# Customer Churn Prediction Project 🔍

## Project Overview
This project focuses on predicting customer churn using machine learning techniques. The goal is to identify customers who are likely to leave the service, enabling businesses to take proactive retention actions and reduce revenue loss.

A Gradient Boosting Classifier was trained and evaluated, and the final model was deployed using Streamlit to provide an interactive prediction interface.

---

## 🎯 Business Objective
- Predict whether a customer is likely to churn (leave the service).  
- Help businesses target high-risk customers with retention strategies.  
- Reduce customer loss and improve long-term profitability.  

---

## 📁 Dataset Description
- Total samples: 2000 customers  
- Target variable: **Churn**  
  - `0` → Customer stayed  
  - `1` → Customer churned  
- Class distribution:  
  - Not Churn (0): 1593 customers  
  - Churn (1): 407 customers  

⚠️ The dataset is imbalanced, which makes evaluation metrics beyond accuracy very important.

---

## 📊 Exploratory Data Analysis (EDA)
- Majority of customers did not churn, indicating class imbalance.  
- Churn behavior shows stronger patterns in a subset of customers rather than the entire population.  
- This imbalance motivated careful evaluation using **precision, recall, and F1-score**.  

*(Detailed visualizations and analysis are included in the Jupyter Notebook.)*

---

## 🛠️ Data Preprocessing
- Encoding categorical variables.  
- Feature scaling using `StandardScaler`.  
- Splitting the dataset into training and testing sets.  
- Saving the scaler for reuse during deployment.  

---

## 🤖 Model Used
- **Algorithm:** Gradient Boosting Classifier  
- **Reason for choice:**  
  - Handles non-linear relationships well.  
  - Performs strongly on structured/tabular data.  
  - Robust against overfitting when tuned properly.  

---

## 📈 Model Evaluation

### ✅ Accuracy
**Accuracy = 0.8665 (~87%)**

### 📊 Classification Report

| Class       | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|--------|
| Not Churn (0) | 0.88      | 0.96   | 0.92     | 1593   |
| Churn (1)    | 0.77      | 0.49   | 0.60     | 407    |
| **Macro Avg** | 0.83      | 0.73   | 0.76     | 2000   |
| **Weighted Avg** | 0.86  | 0.87   | 0.85     | 2000   |

**Interpretation:**  
- The model performs very well in identifying customers who will stay.  
- Recall for churned customers is lower, meaning some churn cases were missed.  
- This trade-off is common in imbalanced datasets.

---

### 🔎 Confusion Matrix Analysis

|                | Predicted Stay | Predicted Churn |
|----------------|----------------|----------------|
| Actual Stay    | 1535 (TN)      | 58 (FP)        |
| Actual Churn   | 209 (FN)       | 198 (TP)       |

- **True Negatives (1535):** Correctly predicted customers who stayed.  
- **True Positives (198):** Correctly identified customers who churned.  
- **False Positives (58):** Predicted churn, but customer actually stayed.  
- **False Negatives (209):** Missed churned customers (most critical business risk).  

📌 Reducing **False Negatives** is important to avoid losing high-risk customers.

---

## 🚀 Deployment
The trained model is deployed using **Streamlit** for interactive predictions.


