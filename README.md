# TELECOM CUSTOMER CHURN PREDICTION

## Project Overview

Customer acquisition costs significantly more than retaining an existing customer. In subscription-based industries such as telecommunications, predicting customer churn is critical for protecting revenue and sustaining long-term growth.

This project builds a machine learning model to predict whether a customer is likely to churn based on demographic information, service usage patterns, billing details, and contract type.

The final selected model is XGBoost, chosen based on its superior recall performance for churn prediction.

---

## Dataset Information

Dataset: Telco Customer Churn  
Total Records: 7,043  
Total Features: 21  

## Target Variable
- Churn  
  - 0 → Customer retained  
  - 1 → Customer churned  

The dataset includes:

- Customer demographics (gender, senior citizen, partner, dependents)
- Account information (tenure, contract type, payment method, billing details)
- Services subscribed (internet service, phone service, tech support, online security, streaming services)

---

## Data Cleaning and Preprocessing

The following preprocessing steps were performed:

- Removed irrelevant identifier column (`customerID`)
- Converted `TotalCharges` to numeric datatype
- Handled 11 missing values
- Encoded categorical variables using LabelEncoder
- Standardized numerical features using StandardScaler
- Performed stratified train-test split to preserve class distribution

These steps ensured consistent, clean, and model-ready data.

---

## Exploratory Data Analysis (EDA) – Key Insights

- Customers with month-to-month contracts show the highest churn rates.
- Fiber optic users exhibit higher churn compared to DSL users.
- Customers without Tech Support or Online Security are more likely to churn.
- High Monthly Charges increase churn probability.
- Customers with lower tenure churn more frequently.
- Electronic check payment method is associated with higher churn.
- Gender has minimal impact on churn behavior.

These insights helped guide model interpretation and feature importance analysis.

---

## Model Training and Evaluation

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- XGBoost

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Why Recall Is More Important Than Accuracy in Churn Prediction

In churn prediction, the primary objective is to identify customers who are likely to leave.

Accuracy alone can be misleading in imbalanced datasets. For example, if most customers do not churn, a model predicting “No Churn” for all customers may achieve high accuracy but fail to detect actual churners.

Recall (for the churn class) measures:

Out of all actual churn customers, how many were correctly identified?

High recall ensures:

- Fewer churners are missed
- High-risk customers are identified early
- Retention strategies can be applied proactively
- Revenue loss is minimized

Since the business goal is to reduce customer attrition, maximizing recall is more important than maximizing overall accuracy.

---

## Final Model Selection: XGBoost

XGBoost was selected as the final model because:

- It achieved the highest recall for churn customers (81% after threshold adjustment).
- It handled class imbalance effectively using `scale_pos_weight`.
- It provided strong overall classification performance.
- It generated churn probability scores for actionable business insights.

Threshold tuning was applied to further improve recall performance.

---

## Churn Probability Prediction

The model outputs churn probability for each customer.

Example:

Customer 1 → 6.65%  
Customer 2 → 60.71%  
Customer 3 → 0.80%  

Customers with higher churn probability can be targeted using:

- Loyalty programs
- Discount offers
- Contract upgrades
- Service improvements

This enables proactive churn prevention.

---

## Model Deployment

Saved artifacts:

- `final_churn_model.pkl` → Trained XGBoost model
- `threshold.pkl` → Optimized classification threshold

The system is deployment-ready and can be integrated into:

- CRM systems
- Retention dashboards
- Automated customer monitoring pipelines

---

## Conclusion

This project successfully:

- Cleaned and prepared real-world telecom data
- Conducted detailed exploratory data analysis
- Addressed class imbalance appropriately
- Compared multiple machine learning models
- Selected the final model based on business-relevant metrics
- Generated churn probability predictions
- Saved deployment-ready model artifacts

The final system enables proactive identification of high-risk customers and supports data-driven retention strategies.
