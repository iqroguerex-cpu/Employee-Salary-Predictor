import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Employee Salary AI", page_icon="💼", layout="wide")

# =========================
# THEME TOGGLE
# =========================
dark_mode = st.sidebar.toggle("🌗 Dark Mode")

if dark_mode:
    sns.set_style("darkgrid")
else:
    sns.set_style("whitegrid")

# =========================
# LOAD DATA
# =========================
dataset = pd.read_csv("Employee_Salary.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# =========================
# PREPROCESSING
# =========================
num_imputer = SimpleImputer(strategy='mean')
X[:, [1,2,4]] = num_imputer.fit_transform(X[:, [1,2,4]])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[:, [0,3]] = cat_imputer.fit_transform(X[:, [0,3]])

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop="first"), [0,3])],
    remainder='passthrough'
)

X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =========================
# MODELS
# =========================
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Metrics
r2_lin = r2_score(y_test, y_pred_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)
r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)

# =========================
# HEADER
# =========================
st.title("💼 Employee Salary AI Dashboard")
st.caption("Linear vs Polynomial Regression • Advanced ML Pipeline")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Analytics", "📈 Model Insights", "📥 Downloads"])

# =========================
# TAB 1 — PREDICTION
# =========================
with tab1:
    st.subheader("Enter Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox("Education", ["Bachelors", "Masters", "PhD"])
        experience = st.slider("Experience (Years)", 0, 20, 5)
        age = st.slider("Age", 20, 65, 30)

    with col2:
        department = st.selectbox("Department", ["IT", "Finance", "HR"])
        performance = st.slider("Performance Score", 1, 5, 3)
        model_choice = st.radio("Select Model", ["Linear", "Polynomial"])

    input_df = pd.DataFrame([[education, experience, age, department, performance]],
                            columns=["Education","Experience","Age","Department","PerformanceScore"])

    input_df.iloc[:, [1,2,4]] = num_imputer.transform(input_df.iloc[:, [1,2,4]])
    input_df.iloc[:, [0,3]] = cat_imputer.transform(input_df.iloc[:, [0,3]])

    input_X = ct.transform(input_df)
    input_X = sc.transform(input_X)

    if st.button("Predict Salary"):
        if model_choice == "Linear":
            prediction = lin_model.predict(input_X)
        else:
            prediction = poly_model.predict(poly.transform(input_X))

        st.success(f"💰 Estimated Salary: ${prediction[0]:,.0f}")

# =========================
# TAB 2 — ANALYTICS
# =========================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(dataset["Salary"], kde=True, ax=ax1)
        ax1.set_title("Salary Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x="Experience", y="Salary", data=dataset, ax=ax2)
        ax2.set_title("Experience vs Salary")
        st.pyplot(fig2)

    st.subheader("Education vs Salary")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Education", y="Salary", data=dataset, ax=ax3)
    st.pyplot(fig3)

# =========================
# TAB 3 — MODEL INSIGHTS
# =========================
with tab3:
    st.metric("Linear R²", f"{r2_lin:.2f}")
    st.metric("Polynomial R²", f"{r2_poly:.2f}")
    st.metric("Linear MAE", f"${mae_lin:,.0f}")
    st.metric("Polynomial MAE", f"${mae_poly:,.0f}")

    st.subheader("Actual vs Predicted (Linear)")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred_lin, ax=ax4)
    ax4.set_xlabel("Actual")
    ax4.set_ylabel("Predicted")
    st.pyplot(fig4)

    st.subheader("Feature Importance (Linear Coefficients)")
    coefficients = lin_model.coef_
    st.write(coefficients)

# =========================
# TAB 4 — DOWNLOADS
# =========================
with tab4:
    results_df = pd.DataFrame({
        "Actual Salary": y_test,
        "Predicted Salary": y_pred_lin
    })

    csv = results_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name="salary_predictions.csv",
        mime="text/csv",
    )
