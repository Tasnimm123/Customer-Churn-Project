# ==============================
# Bank Customer Churn App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Bank Customer Churn",
    page_icon="🏦",
    layout="wide"
)

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Bank Customer Churn Prediction.csv")

df = load_data()

# ------------------------------
# Load Model (Safe Way)
# ------------------------------
def load_model():
    model_path = "model.pkl"

    if not os.path.exists(model_path):
        st.error("❌ model.pkl file not found in project folder.")
        st.stop()

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model

model = load_model()

# ------------------------------
# Project Title
# ------------------------------
st.title("🏦 Bank Customer Churn Prediction System")

st.markdown("""
## 📌 Project Summary

This project predicts whether a bank customer will leave the bank (Churn) or stay.

The model is built using Machine Learning and trained on:
- Customer demographics
- Account information
- Financial features

It helps the bank identify high-risk customers and improve retention strategies.
""")

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2 = st.tabs(["📊 Data Analysis", "🔮 Prediction Page"])

# =====================================================
# TAB 1 — DATA ANALYSIS
# =====================================================
with tab1:

    st.header("📊 Data Analysis & Statistical Summary")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Description")
    st.dataframe(df.describe())

    # Extra Statistics
    avg_salary = df["EstimatedSalary"].mean()
    median_salary = df["EstimatedSalary"].median()
    churn_rate = df["Exited"].mean()
    RPI = (1 - churn_rate) * 100

    st.subheader("Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Salary", f"{avg_salary:,.2f}")
    col2.metric("Median Salary", f"{median_salary:,.2f}")
    col3.metric("Retention Performance Index (RPI)", f"{RPI:.2f}%")

    # ------------------ Plots ------------------

    st.subheader("Churn Distribution")

    fig1, ax1 = plt.subplots()
    sns.countplot(x="Exited", data=df, ax=ax1)
    st.pyplot(fig1)

    st.markdown("This chart shows the number of customers who stayed vs left.")

    st.subheader("Age Distribution by Churn")

    fig2, ax2 = plt.subplots()
    sns.histplot(df, x="Age", hue="Exited", kde=True, ax=ax2)
    st.pyplot(fig2)

    st.markdown("Older customers show higher churn probability.")

    st.subheader("Balance vs Churn")

    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Exited", y="Balance", data=df, ax=ax3)
    st.pyplot(fig3)

    st.markdown("Customers with higher balances may have different churn behavior.")

# =====================================================
# TAB 2 — PREDICTION PAGE
# =====================================================
with tab2:

    st.header("🔮 Predict Customer Churn")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 600)
        age = st.number_input("Age", 18, 100, 30)
        tenure = st.number_input("Tenure (Years)", 0, 20, 5)
        balance = st.number_input("Balance", 0.0, 300000.0, 0.0)

    with col2:
        num_products = st.number_input("Number of Products", 1, 4, 1)
        has_card = st.selectbox("Has Credit Card", [0, 1])
        is_active = st.selectbox("Is Active Member", [0, 1])
        estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    geography = st.selectbox("Geography", df["Geography"].unique())
    gender = st.selectbox("Gender", df["Gender"].unique())

    # Encoding (safe dynamic encoding)
    geo_map = {g: i for i, g in enumerate(df["Geography"].unique())}
    gender_map = {g: i for i, g in enumerate(df["Gender"].unique())}

    geo_encoded = geo_map[geography]
    gender_encoded = gender_map[gender]

    input_data = np.array([[credit_score,
                            geo_encoded,
                            gender_encoded,
                            age,
                            tenure,
                            balance,
                            num_products,
                            has_card,
                            is_active,
                            estimated_salary]])

    if st.button("Predict"):

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = 0

        if prediction == 1:
            st.error(f"⚠️ Customer is likely to leave (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Customer is likely to stay (Probability: {probability:.2f})")

    st.markdown("""
    ---
    ### 📚 About This Project
    - Built with Python & Streamlit
    - Machine Learning Classification Model
    - Includes Data Analysis & Business Metrics
    - Developed for academic purposes
    """)
