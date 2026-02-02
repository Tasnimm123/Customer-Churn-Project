# 📊 Customer Churn Prediction Project

## 🔍 Project Overview
This project aims to predict customer churn using machine learning techniques on a structured banking dataset.  
Customer churn prediction helps businesses identify customers who are at risk of leaving and take proactive retention actions.

The project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Handling imbalanced data
- Training and evaluating multiple machine learning models
- Deploying the final model using Streamlit

---

## 🎯 Business Objective
- Predict whether a customer will churn or not.
- Reduce customer loss by identifying high-risk customers early.
- Support data-driven decision-making for customer retention strategies.

---

## 📁 Dataset Description
- **Total records:** 10,000 customers
- **Target variable:** `churn`
  - `0` → Customer stayed
  - `1` → Customer churned

### Features:
- `credit_score`
- `country`
- `gender`
- `age`
- `tenure`
- `balance`
- `products_number`
- `credit_card`
- `active_member`
- `estimated_salary`

The dataset contains **no missing values**, and data types include numerical and categorical features.

---

## 📊 Exploratory Data Analysis (EDA)
Key observations from EDA:
- The dataset is **imbalanced**:
  - Not Churn (0): 7,963 customers
  - Churn (1): 2,037 customers
- This imbalance makes accuracy an unreliable metric on its own.
- Initial analysis explored feature distributions and their relationship with churn.

---

## 🚨 Outlier Analysis
Outlier detection was performed using the **Interquartile Range (IQR)** method:

- **Age:** 359 outliers were detected, mainly representing senior customers.
- **Balance:** No extreme outliers beyond logical banking limits were found.

📌 **Decision:**  
Outliers were retained because extreme values may carry important information for churn prediction, especially in identifying high-risk customer segments.

---

## 🛠️ Data Preprocessing
The following preprocessing steps were applied:
- Encoding categorical variables.
- Feature scaling using `StandardScaler`.
- Train-test split.
- Handling class imbalance using **SMOTE**, applied **only to the training data** to prevent data leakage.

---

## ⚖️ Handling Imbalanced Data
The dataset suffers from class imbalance, which negatively impacts churn detection.

To address this:
- **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training set.
- This improves the model’s ability to learn churn patterns and increases recall for the minority class.

📌 In churn prediction, improving recall is more important than maximizing accuracy.

---

## 🤖 Models Used
Multiple machine learning models were trained and evaluated, including:
- Logistic Regression
- Random Forest
- Gradient Boosting Classifier

The final model selection was based on **recall and F1-score**, not accuracy alone.

---

## 📈 Model Evaluation (Gradient Boosting)

### Accuracy
Accuracy: 0.8665 (~87%)


### Classification Report

| Class | Precision | Recall | F1-score | Support |
|------|----------|--------|----------|---------|
| Not Churn (0) | 0.88 | 0.96 | 0.92 | 1593 |
| Churn (1) | 0.77 | 0.49 | 0.60 | 407 |
| **Macro Avg** | 0.83 | 0.73 | 0.76 | 2000 |
| **Weighted Avg** | 0.86 | 0.87 | 0.85 | 2000 |

---

## 🧮 Confusion Matrix Analysis
- **True Negatives (1535):** Correctly predicted customers who stayed.
- **True Positives (198):** Correctly identified churned customers.
- **False Positives (58):** Customers predicted to churn but actually stayed.
- **False Negatives (209):** Churned customers that the model failed to identify.

📌 Reducing false negatives is crucial, as missed churned customers directly impact business revenue.

---

## 📊 Evaluation Metrics Explanation
- **Precision:** Measures how many customers predicted as churn actually churned.
- **Recall:** Measures how many actual churned customers were correctly identified.
- **F1-score:** Balances precision and recall and is especially useful for imbalanced datasets.

Due to class imbalance, **recall and F1-score were prioritized over accuracy** when evaluating model performance.

---

## 🚀 Deployment
The final model was deployed using **Streamlit** to allow interactive churn prediction.

Notes:
1️⃣ Accuracy is not sufficient for churn prediction due to class imbalance.
2️⃣ Recall is more important because missing churned customers leads to business loss.
3️⃣ SMOTE was applied only on training data to avoid data leakage.
4️⃣ Gradient Boosting performs well on structured tabular data.
5️⃣ The model was deployed using Streamlit to make it accessible




