import streamlit as st
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Load model & threshold
model = XGBClassifier()
model.load_model("model/xgb_churn_model.json")

with open("model/config.json") as f:
    threshold = json.load(f)["threshold"]

st.title("Bank Customer Churn Prediction")

FEATURE_COLUMNS = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
]

# User inputs
credit_score = st.slider("Credit Score", 300, 850, 650)
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure (Years)", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
num_products = st.slider("Number of Products", 1, 4, 2)
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

is_active = st.selectbox("Active Member", [0, 1])
has_card = st.selectbox("Has Credit Card", [0, 1])

gender = st.radio("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# Encode gender
gender_val = 1 if gender == "Male" else 0

# Encode geography (one-hot)
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# Build input dataframe
input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Gender": gender_val,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": estimated_salary,
    "Geography_Germany": geo_germany,
    "Geography_Spain": geo_spain,
}])[FEATURE_COLUMNS]


if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]
    prediction = int(prob >= threshold)

    st.write(f"### Churn Probability: {prob:.2f}")

    if prediction == 1:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")
