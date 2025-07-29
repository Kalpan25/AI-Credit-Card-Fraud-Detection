# Streamlit Dashboard

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("random_forest_model.pkl")
# model = joblib.load("logistic_regression_model.pkl")

st.title("Credit Card Fraud Detection")
amount = st.number_input("Enter Scaled Amount:")
time = st.number_input("Enter Scaled Time:")
# Add all other features in real use case

if st.button("Predict"):
    # Dummy input for now
    input_data = [[time, amount] + [0]*28]
    prediction = model.predict(input_data)
    st.write("Prediction:", "Fraud" if prediction[0] == 1 else "Legit")
