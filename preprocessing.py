# Importing Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

try:
    import streamlit as st
    STREAMLIT = True
except ImportError:
    STREAMLIT = False

# Load Dataset
df = pd.read_csv("data/creditcard.csv")
print("Data loaded Ready for use!")

# View first few rows
print(df.head())

# Dataset shape and class distribution
print("Shape of the dataset:", df.shape)
print("\nClass distribution (0 = Legit, 1 = Fraud):")
print(df['Class'].value_counts(normalize=True))

# Visualize class imbalance
sns.countplot(data=df, x="Class")
plt.title("Class Distribution: Fraud (1) vs Legit (0)")
if STREAMLIT:
    st.pyplot(plt)
else:
    plt.show()

# Feature Scaling
scaler = StandardScaler()
df["Scaled_Time"] = scaler.fit_transform(df["Time"].values.reshape(-1 , 1))
df["Scaled_Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1 , 1))

# Drop original columns to avoid duplication
df.drop(['Time', 'Amount'], axis=1, inplace=True)
print(df.head())


def load_and_preprocess_data(path="data/creditcard.csv"):
    df = pd.read_csv(path)
    scaler = StandardScaler()
    df["Scaled_Time"] = scaler.fit_transform(df["Time"].values.reshape(-1 , 1))
    df["Scaled_Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1 , 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df





