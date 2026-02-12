# 📊 Customer Churn Prediction

## 📌 Project Overview
This project aims to predict customer churn using Machine Learning techniques.  
Customer churn prediction is a critical business problem, as retaining existing customers is more cost-effective than acquiring new ones.

The goal of this project is to identify customers who are likely to leave the company, enabling proactive retention strategies.

---

## 🎯 Business Objective

In churn prediction, the primary objective is:

> **Identifying as many churned customers as possible.**

For this reason, this project prioritizes **Recall** over Accuracy.

Missing a churned customer (**False Negative**) has a higher business cost than incorrectly predicting churn (**False Positive**).

---

## 📂 Dataset Description

The dataset includes the following customer attributes:

- Credit Score  
- Geography (Country)  
- Gender  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Has Credit Card  
- Is Active Member  
- Estimated Salary  

**Target Variable:**
- `Exited`
  - `1` → Customer Churned  
  - `0` → Customer Stayed  

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Checked for missing values
- Encoded categorical variables using `OneHotEncoder` (scikit-learn)
- Split data into training and testing sets
- Applied feature scaling using `StandardScaler`

---

### 2️⃣ Handling Imbalanced Data
The dataset was imbalanced.  
To address this issue:

- Applied **SMOTE (Synthetic Minority Oversampling Technique)**
- SMOTE was applied **only on training data** to prevent data leakage

---

### 3️⃣ Modeling

The following models were trained and evaluated:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  

After comparison, **Gradient Boosting** achieved the best performance.

---

### 4️⃣ Model Evaluation

The model was evaluated using:

- Confusion Matrix  
- Accuracy  
- Precision  
- Recall  
- F1-Score  

However, since the business objective is to detect churned customers, **Recall was considered the most important metric**.

---

## 📈 ROC Curve & Threshold Optimization

Instead of relying on the default classification threshold (0.5), the ROC Curve was used to:

- Analyze performance across different thresholds
- Calculate AUC Score
- Select an optimal threshold using Youden’s J statistic

This approach helped improve Recall performance.

---

## ✅ Final Model Performance

After ROC-based threshold tuning:

- 🎯 **Recall: 79%**
- Accuracy slightly decreased but remained acceptable
- The model successfully identifies the majority of churn customers

This aligns with the business goal of minimizing customer loss.

---

## 🧠 Why Recall Matters More Than Accuracy

In churn prediction:

- **False Negative** → Customer leaves without being detected ❌ (High business impact)  
- **False Positive** → Customer predicted to churn but stays ⚠ (Lower impact)  

Therefore:

> Optimizing Recall is more important than maximizing Accuracy in this project.

---

## 🛠 Tools & Technologies

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Imbalanced-learn (SMOTE)  

---

## 🚀 Key Takeaways

- Handling imbalanced data significantly improves model performance.
- Accuracy alone is not reliable in imbalanced classification problems.
- ROC Curve helps optimize classification threshold.
- Business understanding is essential when evaluating ML models.

---

## 📌 Conclusion

This project demonstrates how Machine Learning can be applied to solve real-world business problems.

By prioritizing **Recall (79%)** instead of Accuracy, the final model better supports the business objective of reducing customer churn and minimizing revenue loss.


Notes:
1️⃣ Accuracy is not sufficient for churn prediction due to class imbalance.
2️⃣ Recall is more important because missing churned customers leads to business loss.
3️⃣ SMOTE was applied only on training data to avoid data leakage.
4️⃣ Gradient Boosting performs well on structured tabular data.
5️⃣ The model was deployed using Streamlit to make it accessible




