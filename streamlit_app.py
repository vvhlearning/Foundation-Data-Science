import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load('Live_Disease_Prediction_RF_Model.pkl')  # Replace with your actual model file

st.title("Liver Disease Prediction App")

st.markdown("Enter patient data below to predict liver disease:")

# Create input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ['Male', 'Female'])

direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, step=0.1)
alk_phos = st.number_input("Alkaline Phosphotase", min_value=0.0, step=1.0)
ast = st.number_input("Aspartate Aminotransferase", min_value=0.0, step=1.0)
agr = st.number_input("Albumin and Globulin Ratio", min_value=0.0, step=0.1)

# Convert gender to numeric if required by model
gender_val = 1 if gender == 'Male' else 0

# Predict on button click
if st.button("Predict"):
    # Create feature vector
    features = np.array([[age, gender_val, direct_bilirubin, alk_phos, ast, agr]])

    # Make prediction
    prediction = model.predict(features)

    # Display result
    if prediction[0] == 1:
        st.error("The model predicts the patient is at risk of liver disease.")
    else:
        st.success("The model predicts the patient is not at risk of liver disease.")
