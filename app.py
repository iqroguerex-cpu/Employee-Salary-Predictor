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

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide"
)

# ==============================
# Custom CSS Styling
# ==============================
st.markdown("""
<style>
.main-title {
    font-size:40px;
    font-weight:700;
}
.metric-box {
    background-color:#f0f2f6;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Data
# ==============================
dataset = pd.read_csv("Employee_Salary.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ==============================
# Preprocessing
# ==============================
num_imputer = SimpleImputer(strategy='mean')
X[:, [1,2,4]] = num_imputer.fit_transform(X[:, [1,2,4]])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[:, [0,3]] = cat_imputer.fit_transform(X[:, [0,3]])

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop="first"), [0,3])],
    remainder='passthrough'
)

X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ==============================
# Models
# ==============================
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

r2_lin = r2_score(y_test, y_pred_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)

r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)

# ==============================
# UI Layout
# ==============================

st.markdown('<div class="main-title">💼 Employee Salary Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown("Predict employee salary using advanced regression models.")

st.divider()

# Sidebar
st.sidebar.header("🧾 Employee Details")

education = st.sidebar.selectbox("Education", ["Bachelors", "Masters", "PhD"])
experience = st.sidebar.slider("Experience (Years)", 0, 20, 5)
age = st.sidebar.slider("Age", 20, 65, 30)
department = st.sidebar.selectbox("Department", ["IT", "Finance", "HR"])
performance = st.sidebar.slider("Performance Score", 1, 5, 3)

model_choice = st.sidebar.radio("Model", ["Linear Regression", "Polynomial Regression"])

input_df = pd.DataFrame(
    [[education, experience, age, department, performance]],
    columns=["Education","Experience","Age","Department","PerformanceScore"]
)

input_df.iloc[:, [1,2,4]] = num_imputer.transform(input_df.iloc[:, [1,2,4]])
input_df.iloc[:, [0,3]] = cat_imputer.transform(input_df.iloc[:, [0,3]])

input_X = ct.transform(input_df)
input_X = sc.transform(input_X)

if st.sidebar.button("🔮 Predict Salary"):
    if model_choice == "Linear Regression":
        prediction = lin_model.predict(input_X)
    else:
        prediction = poly_model.predict(poly.transform(input_X))

    st.success(f"💰 Estimated Salary: ${prediction[0]:,.0f}")

st.divider()

# ==============================
# Metrics Section
# ==============================

col1, col2 = st.columns(2)

with col1:
    st.metric("Linear R² Score", f"{r2_lin:.2f}")
    st.metric("Linear MAE", f"${mae_lin:,.0f}")

with col2:
    st.metric("Polynomial R² Score", f"{r2_poly:.2f}")
    st.metric("Polynomial MAE", f"${mae_poly:,.0f}")

st.divider()

# ==============================
# Charts Section
# ==============================

sns.set_style("whitegrid")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(dataset["Salary"], kde=True, color="#4CAF50", ax=ax1)
    ax1.set_title("Salary Distribution", fontsize=14)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x="Experience", y="Salary", data=dataset, color="#2196F3", ax=ax2)
    ax2.set_title("Experience vs Salary", fontsize=14)
    st.pyplot(fig2)

st.subheader("🎓 Education Impact on Salary")
fig3, ax3 = plt.subplots()
sns.boxplot(x="Education", y="Salary", data=dataset, palette="Set2", ax=ax3)
st.pyplot(fig3)

st.subheader("📉 Actual vs Predicted (Linear Model)")
fig4, ax4 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred_lin, color="#FF5722", ax=ax4)
ax4.set_xlabel("Actual Salary")
ax4.set_ylabel("Predicted Salary")
st.pyplot(fig4)
