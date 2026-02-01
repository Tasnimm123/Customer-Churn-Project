import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the pre-trained model and scaler
# Make sure model.pkl and scaler.pkl are in the same directory as this script
model = joblib.load('gradient_boosting_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Set App Title and Description
st.title("Customer Churn Prediction App")
st.write("Enter the customer details below to predict the likelihood of leaving the bank.")

# 3. Create input fields for the 11 features (matching X_train.shape)
# Splitting the inputs into two columns for better UI layout
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", value=600)
    age = st.number_input("Age", value=30)
    tenure = st.number_input("Tenure (Years)", value=5)
    balance = st.number_input("Balance", value=0.0)

with col2:
    num_products = st.number_input("Number of Products", value=1)
    has_crcard = st.selectbox("Has Credit Card? (0=No, 1=Yes)", [0, 1])
    is_active = st.selectbox("Is Active Member? (0=No, 1=Yes)", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# Add Geography and Gender select boxes
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

# 4. Prediction Logic
if st.button("Predict"):
    # Manual encoding to match the 11-column structure used during training
    # Handling Geography (One-Hot Encoding style)
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0
    
    # Handling Gender (Binary Encoding style)
    gender_male = 1 if gender == "Male" else 0
    
    # Organize features into a numpy array (must follow the exact order of training columns)
    features = np.array([[credit_score, age, tenure, balance, num_products, 
                          has_crcard, is_active, estimated_salary, 
                          geo_germany, geo_spain, gender_male]])
    
    # Apply the same scaling transformation used during training
    features_scaled = scaler.transform(features)
    
    # Perform the prediction
    prediction = model.predict(features_scaled)
    
    # Display the result based on the prediction (1 = Churn, 0 = Not Churn)
    if prediction[0] == 1:
        st.error("Prediction: The customer is likely to CHURN (Leave the bank)")
    else:
        st.success("Prediction: The customer is likely to STAY (Not Churn)")
      

