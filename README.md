
📊 Customer Churn Prediction Project
📌 Project Overview

This project aims to predict customer churn using Machine Learning techniques.
Customer churn prediction is a critical business problem, as retaining existing customers is more cost-effective than acquiring new ones.

The main objective of this project is to identify customers who are likely to leave the company, allowing the business to take proactive retention actions.

🎯 Business Objective

In churn prediction, the most important goal is:

Identifying as many churned customers as possible.

For this reason, this project focuses more on Recall rather than Accuracy.

Missing a churned customer (False Negative) is more costly than incorrectly predicting that a customer will churn.

📂 Dataset Description

The dataset contains customer information such as:

Credit Score

Geography (Country)

Gender

Age

Tenure

Balance

Number of Products

Has Credit Card

Is Active Member

Estimated Salary

Target Variable:

Exited

1 → Customer Churned

0 → Customer Stayed

⚙️ Project Workflow
1️⃣ Data Preprocessing

Checked for missing values

Encoded categorical variables using OneHotEncoder (sklearn)

Split data into training and testing sets

Applied feature scaling using StandardScaler

2️⃣ Handling Imbalanced Data

The dataset was imbalanced, so I applied:

SMOTE (Synthetic Minority Oversampling Technique)
Applied only on training data to avoid data leakage.

3️⃣ Modeling

The following classification models were tested:

Logistic Regression

Random Forest

Gradient Boosting

After evaluation, Gradient Boosting provided strong performance.

4️⃣ Model Evaluation

The model was evaluated using:

Confusion Matrix

Accuracy

Precision

Recall

F1-Score

However, since the business goal is to detect churn customers, Recall was prioritized over Accuracy.

📈 ROC Curve & Threshold Optimization

Instead of using the default classification threshold (0.5), I used:

ROC Curve

AUC Score

Threshold Optimization (Youden’s J statistic)

This helped select a better threshold that improves Recall performance.

✅ Final Model Performance

After applying ROC-based threshold tuning:

🎯 Recall: 79%

Accuracy decreased slightly but remained acceptable

The model successfully identifies most churn customers

This result aligns with the business objective, where detecting churn is more important than maximizing overall accuracy.

🧠 Why Recall Matters More Than Accuracy

In churn prediction:

False Negative → A customer leaves and we fail to detect it ❌ (High business cost)

False Positive → We predict churn but customer stays ⚠ (Lower cost)

Therefore:

Improving Recall is more important than maximizing Accuracy in this project.

🛠 Tools & Libraries

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Imbalanced-learn (SMOTE)

🚀 Key Takeaways

Handling imbalanced data significantly improves model performance.

Accuracy alone is not reliable for imbalanced classification problems.

ROC Curve helps optimize classification threshold.

Business understanding is essential when evaluating ML models.

📌 Conclusion

This project demonstrates how Machine Learning can solve real business problems.
By focusing on Recall (79%) instead of Accuracy, the final model better serves the business objective of minimizing customer churn and reducing revenue loss.

Notes:
1️⃣ Accuracy is not sufficient for churn prediction due to class imbalance.
2️⃣ Recall is more important because missing churned customers leads to business loss.
3️⃣ SMOTE was applied only on training data to avoid data leakage.
4️⃣ Gradient Boosting performs well on structured tabular data.
5️⃣ The model was deployed using Streamlit to make it accessible




