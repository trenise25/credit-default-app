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
    SEX = st.selectbox("Gender (1=Male, 2=Female)", [1, 2])
    EDUCATION = st.selectbox("Education (1=Grad, 2=Univ, 3=HighSchool, 4=Other)", [1, 2, 3, 4])
    MARRIAGE = st.selectbox("Marriage (1=Married, 2=Single, 3=Other)", [1, 2, 3])
    PAY_0 = st.slider("Repayment Status in September (PAY_0)", -2, 8, value=0)
    PAY_2 = st.slider("Repayment Status in August (PAY_2)", -2, 8, value=0)
    PAY_3 = st.slider("Repayment Status in July (PAY_3)", -2, 8, value=0)
    PAY_4 = st.slider("Repayment Status in June (PAY_4)", -2, 8, value=0)
    PAY_5 = st.slider("Repayment Status in May (PAY_5)", -2, 8, value=0)
    PAY_6 = st.slider("Repayment Status in April (PAY_6)", -2, 8, value=0)
    
    BILL_AMT1 = st.number_input("Bill Amount for September", value=5000)
    BILL_AMT2 = st.number_input("Bill Amount for August", value=4000)
    BILL_AMT3 = st.number_input("Bill Amount for July", value=3000)
    BILL_AMT4 = st.number_input("Bill Amount for June", value=2000)
    BILL_AMT5 = st.number_input("Bill Amount for May", value=1000)
    BILL_AMT6 = st.number_input("Bill Amount for April", value=500)

    PAY_AMT1 = st.number_input("Payment Amount in September", value=1000)
    PAY_AMT2 = st.number_input("Payment Amount in August", value=800)
    PAY_AMT3 = st.number_input("Payment Amount in July", value=600)
    PAY_AMT4 = st.number_input("Payment Amount in June", value=400)
    PAY_AMT5 = st.number_input("Payment Amount in May", value=200)
    PAY_AMT6 = st.number_input("Payment Amount in April", value=100)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
                            PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                            BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
                            PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]])

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.subheader("📊 Prediction Result")
    st.write("Prediction:", "❌ Will Default" if prediction else "✅ No Default")
    st.write(f"Default Risk Probability: **{prob:.2%}**")
