import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Loan Default Predictor", layout="centered")


# 💾 Load & preprocess your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Harri/OneDrive/Desktop/JN/Machine_Learning/MLassignment2/Assignment 2/df1_loan.csv")

    df.fillna({
        'Gender': df['Gender'].mode()[0],
        'Married': df['Married'].mode()[0],
        'Dependents': df['Dependents'].mode()[0],
        'Self_Employed': df['Self_Employed'].mode()[0],
        'LoanAmount': df['LoanAmount'].median(),
        'Loan_Amount_Term': df['Loan_Amount_Term'].mode()[0],
        'Credit_History': df['Credit_History'].mode()[0]
    }, inplace=True)

    df['Total_Income'] = df['Total_Income'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    df['Total_Income'] = df['Total_Income'].astype(float)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Married'] = le.fit_transform(df['Married'])
    df['Self_Employed'] = le.fit_transform(df['Self_Employed'])
    df = pd.get_dummies(df, columns=['Dependents', 'Education', 'Property_Area'], drop_first=True)

    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

# 🧠 Train all three models
@st.cache_resource
def train_models():
    df = load_data()
    X = df.drop(columns=['Loan_Status', 'Loan_ID'])
    y = LabelEncoder().fit_transform(df['Loan_Status'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear"),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models, X_train

models, X_train = train_models()

# 🎨 Streamlit UI

st.markdown("## 💸 Loan Default Risk Predictor")
st.caption("This app uses a machine learning model trained on real loan data to predict loan default risk.")

st.markdown("---")
st.subheader("📋 Choose Model")
model_name = st.selectbox("Select a model", list(models.keys()))
model = models[model_name]

st.subheader("🧾 Enter Borrower Information")

loan_amnt = st.slider("Loan Amount", 100, 700, 250)
app_income = st.slider("Applicant Income", 0, 100000, 5000)
coapp_income = st.slider("Coapplicant Income", 0, 50000, 2000)
total_income = st.slider("Total Income ($)", 0, 200000, 10000)
credit_history = st.selectbox("Credit History (1 = good, 0 = bad)", [1.0, 0.0])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])

# Manual encoding to match training
input_data = {
    'LoanAmount': loan_amnt,
    'ApplicantIncome': app_income,
    'CoapplicantIncome': coapp_income,
    'Total_Income': total_income,
    'Credit_History': credit_history,
    'Gender': 1 if gender == "Male" else 0,
    'Married': 1 if married == "Yes" else 0,
    'Self_Employed': 1 if self_employed == "Yes" else 0,
    'Dependents_1': 1 if dependents == '1' else 0,
    'Dependents_2': 1 if dependents == '2' else 0,
    'Dependents_3+': 1 if dependents == '3+' else 0,
    'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
    'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
    'Property_Area_Urban': 1 if property_area == "Urban" else 0
}

# Ensure all features exist
for col in X_train.columns:
    if col not in input_data:
        input_data[col] = 0

input_df = pd.DataFrame([input_data])
input_df = input_df[X_train.columns]  # match column order

if st.button("🚀 Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High risk of default ({proba:.0%} chance)")
    else:
        st.success(f"✅ Likely to repay the loan ({(1 - proba):.0%} confidence)")

    st.caption(f"Model used: **{model_name}**")
