import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/salary_model.joblib")

st.title("💼 Employee Salary Prediction")
st.markdown("Enter the employee details below:")

# Create two columns for input layout
col1, col2 = st.columns(2)

# Column 1 Inputs
with col1:
    age = st.number_input("Age", 18, 90, 30)
    workclass = st.number_input("Workclass (Encoded)", 0, 8, 4)
    education = st.slider("Educational-Num", 1, 16, 10)
    marital_status = st.number_input("Marital Status (Encoded)", 0, 6, 2)
    occupation = st.number_input("Occupation (Encoded)", 0, 14, 3)

# Column 2 Inputs
with col2:
    gender = st.number_input("Gender (0=Female, 1=Male)", 0, 1, 1)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 10000, 0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.number_input("Native Country (Encoded)", 0, 40, 0)

# Prepare input as DataFrame with correct column names
input_data = pd.DataFrame([{
    "age": age,
    "workclass": workclass,
    "educational-num": education,
    "marital-status": marital_status,
    "occupation": occupation,
    "gender": gender,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": native_country
}])

# Ensure column order matches training
input_data = input_data[model.feature_names_in_]

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.subheader("🧠 Prediction Result")
    st.write(f"**Predicted Income Class:** `{prediction}`")

    st.subheader("📊 Prediction Probabilities")
    st.write(f"<=50K: {prob[0]:.4f}, >50K: {prob[1]:.4f}")

    st.subheader("🧪 Input Sent to Model")
    st.dataframe(input_data)
