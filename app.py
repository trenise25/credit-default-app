import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Credit Default Risk", layout="centered")
st.title("💳 Credit Default Risk Predictor")
st.write("This app uses financial and behavioral inputs to predict the probability of a client defaulting on a credit card loan.")

# Load model and scaler
model = joblib.load("credit_default_xgb.pkl")
scaler = joblib.load("scaler.pkl")

# Input form
with st.form("credit_form"):
    LIMIT_BAL = st.number_input("Credit Limit (LIMIT_BAL)", value=20000)
    AGE = st.number_input("Age", min_value=18, max_value=100, value=30)
    SEX = st.selectbox("Gender (SEX)", [1, 2])
    EDUCATION = st.selectbox("Education (1=Grad, 2=Univ, 3=HighSchool, 4=Other)", [1, 2, 3, 4])
    MARRIAGE = st.selectbox("Marriage (1=Married, 2=Single, 3=Other)", [1, 2, 3])
    PAY_0 = st.slider("Last Repayment Status (PAY_0)", -2, 8, value=0)
    BILL_AMT1 = st.number_input("Last Month Bill Amount (BILL_AMT1)", value=5000)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
                            PAY_0, 0, 0, 0, 0, 0, 0, BILL_AMT1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0]])

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.subheader("📊 Prediction Result")
    st.write("Prediction:", "❌ Will Default" if prediction else "✅ No Default")
    st.write(f"Default Risk Probability: **{prob:.2%}**")
