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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

# -----------------------------
# Load Dataset
# -----------------------------
dataset = pd.read_csv("Employee_Salary.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# -----------------------------
# Handle Missing Values
# -----------------------------
num_imputer = SimpleImputer(strategy='mean')
X[:, [1,2,4]] = num_imputer.fit_transform(X[:, [1,2,4]])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[:, [0,3]] = cat_imputer.fit_transform(X[:, [0,3]])

# -----------------------------
# Encode Categorical Columns
# -----------------------------
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop="first"), [0,3])],
    remainder='passthrough'
)

X = ct.fit_transform(X)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# -----------------------------
# Scaling
# -----------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -----------------------------
# Linear Regression
# -----------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

r2_lin = r2_score(y_test, y_pred_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)

# -----------------------------
# Polynomial Regression
# -----------------------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)

# =============================
# STREAMLIT UI
# =============================

st.title("💼 Employee Salary Prediction App")
st.markdown("Predict employee salary using Linear & Polynomial Regression.")

st.sidebar.header("Enter Employee Details")

education = st.sidebar.selectbox("Education", ["Bachelors", "Masters", "PhD"])
experience = st.sidebar.slider("Experience (Years)", 0, 20, 5)
age = st.sidebar.slider("Age", 20, 65, 30)
department = st.sidebar.selectbox("Department", ["IT", "Finance", "HR"])
performance = st.sidebar.slider("Performance Score", 1, 5, 3)

input_df = pd.DataFrame(
    [[education, experience, age, department, performance]],
    columns=["Education","Experience","Age","Department","PerformanceScore"]
)

# Apply same preprocessing
input_df.iloc[:, [1,2,4]] = num_imputer.transform(input_df.iloc[:, [1,2,4]])
input_df.iloc[:, [0,3]] = cat_imputer.transform(input_df.iloc[:, [0,3]])

input_X = ct.transform(input_df)
input_X = sc.transform(input_X)

# Choose Model
model_choice = st.sidebar.radio("Select Model", ["Linear Regression", "Polynomial Regression"])

if st.sidebar.button("Predict Salary"):
    if model_choice == "Linear Regression":
        prediction = lin_model.predict(input_X)
    else:
        input_poly = poly.transform(input_X)
        prediction = poly_model.predict(input_poly)

    st.success(f"💰 Estimated Salary: ${prediction[0]:,.0f}")

# =============================
# VISUALIZATIONS
# =============================

st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

# 1️⃣ Salary Distribution
with col1:
    fig1 = plt.figure()
    plt.hist(dataset["Salary"], bins=10)
    plt.title("Salary Distribution")
    plt.xlabel("Salary")
    plt.ylabel("Frequency")
    st.pyplot(fig1)

# 2️⃣ Experience vs Salary
with col2:
    fig2 = plt.figure()
    sns.scatterplot(x="Experience", y="Salary", data=dataset)
    plt.title("Experience vs Salary")
    st.pyplot(fig2)

# 3️⃣ Education vs Salary
st.subheader("🎓 Education Impact on Salary")
fig3 = plt.figure()
sns.boxplot(x="Education", y="Salary", data=dataset)
st.pyplot(fig3)

# =============================
# MODEL PERFORMANCE
# =============================

st.subheader("📈 Model Performance Comparison")

st.write(f"**Linear Regression R²:** {r2_lin:.2f}")
st.write(f"**Linear Regression MAE:** ${mae_lin:,.0f}")

st.write(f"**Polynomial Regression R²:** {r2_poly:.2f}")
st.write(f"**Polynomial Regression MAE:** ${mae_poly:,.0f}")

# Actual vs Predicted
st.subheader("📉 Actual vs Predicted (Linear Model)")
fig4 = plt.figure()
plt.scatter(y_test, y_pred_lin)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
st.pyplot(fig4)
