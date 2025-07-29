import streamlit as st
from preprocessing import load_and_preprocess_data
from random_forest import run_random_forest
from logistic_regression import run_logistic_regression
from neural_network import run_neural_network
from xgboost_model import run_xgboost
import matplotlib.pyplot as plt
import seaborn as sns

st.title("AI Credit Card Fraud Detection")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Select a model:",
    ("Random Forest", "Logistic Regression", "Neural Network", "XGBoost")
)

# File uploader
uploaded_file = st.file_uploader("Upload your creditcard.csv file", type=["csv"])

df = None
if uploaded_file is not None:
    # Save uploaded file to a temp location and preprocess
    with open("data/creditcard.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    df = load_and_preprocess_data("data/creditcard.csv")
    st.success("Data loaded and preprocessed!")

    # Button to show class distribution plot
    if st.button("Show Class Distribution Plot"):
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Class", ax=ax)
        ax.set_title("Class Distribution: Fraud (1) vs Legit (0)")
        st.pyplot(fig)

    if st.button("Run Model"):
        if model_choice == "Random Forest":
            st.write("Running Random Forest...")
            report = run_random_forest(df)
            st.text(report)
        elif model_choice == "Logistic Regression":
            st.write("Running Logistic Regression...")
            report = run_logistic_regression(df)
            st.text(report)
        elif model_choice == "Neural Network":
            st.write("Running Neural Network...")
            report = run_neural_network(df)
            st.text(report)
        elif model_choice == "XGBoost":
            st.write("Running XGBoost...")
            report = run_xgboost(df)
            st.text(report)
else:
    st.info("Please upload your creditcard.csv file to begin.")