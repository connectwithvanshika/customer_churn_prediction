# Telecom Customer Churn Prediction and Retention Analytics System

## Project Overview

Customer churn is one of the most critical challenges for subscription-based businesses. Studies show that acquiring a new customer costs significantly more than retaining an existing one. Therefore, predicting which customers are likely to leave helps organizations take preventive action and improve retention.

This project builds an end-to-end machine learning system that predicts whether a telecom customer will churn based on their demographics, service usage, contract type, and billing behavior. The final model is deployed through an interactive Streamlit web application that allows real-time churn prediction and probability scoring.

The system focuses on identifying high-risk customers early so that businesses can implement targeted retention strategies.

---

## Problem Statement

Customer churn directly impacts revenue and long-term growth. Businesses need a predictive system that can:

- Identify customers likely to leave
- Analyze key churn-driving factors
- Provide probability-based predictions
- Help teams take proactive retention decisions

The objective of this project is to build a high-recall churn prediction model that accurately detects customers who are likely to churn.

---

## Dataset Information

Dataset used: IBM Telco Customer Churn Dataset

Total records: 7043  
Total features: 21  

The dataset contains:

Customer Demographics:
- Gender
- Senior citizen
- Partner
- Dependents

Customer Account Information:
- Tenure
- Contract type
- Payment method
- Paperless billing
- Monthly charges
- Total charges

Services Subscribed:
- Phone service
- Internet service
- Online security
- Online backup
- Device protection
- Tech support
- Streaming TV and movies

Target Variable:
- Churn (Yes/No)

---

## Data Cleaning and Preprocessing

Key preprocessing steps performed:

1. Removed customerID column (not useful for prediction)
2. Converted TotalCharges to numeric datatype
3. Found 11 missing values due to blank spaces
4. Observed tenure = 0 for those rows
5. Removed rows with zero tenure
6. Filled remaining missing values using mean
7. Converted SeniorCitizen from numeric to categorical
8. Applied label encoding to categorical features
9. Standardized numerical features:
   - tenure
   - MonthlyCharges
   - TotalCharges
10. Stratified train-test split to handle class imbalance

Final dataset ready for model training.

---

## Exploratory Data Analysis (Key Insights)

Major findings from EDA:

- Customers with month-to-month contracts show highest churn
- Long-term contracts significantly reduce churn
- Fiber optic users show higher churn than DSL users
- Customers without tech support and online security churn more
- Electronic check payment method has highest churn rate
- High monthly charges increase churn probability
- Low tenure customers churn more frequently
- Customers with lower total spending churn more
- Gender has minimal impact on churn
- New customers are more likely to leave than loyal customers

Strongest predictors of churn:
- Contract type
- Tenure
- Tech support
- Online security
- Payment method
- Monthly charges

---

## Machine Learning Models Used

Multiple models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost

Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

---

## Final Model Selection

Final chosen model: XGBoost Classifier

Reason:
Although Logistic Regression had slightly higher accuracy (~80%), XGBoost achieved significantly higher recall for churn customers.

Recall for churn customers: ~81%

High recall is important because:
Missing a churn customer means losing a customer without intervention.

Therefore, recall was prioritized over accuracy.

---

## Threshold Optimization

Default threshold: 0.5  
Adjusted threshold: 0.4  

Lowering threshold improved recall and allowed the model to capture more churn-risk customers.

Prediction logic:
If churn probability > 0.4 → Customer likely to churn  
Else → Customer likely to stay  

This makes the model more business-focused and retention-friendly.

---

## Model Performance

Logistic Regression Accuracy: ~80%  
XGBoost Accuracy: ~78–81%  
XGBoost Recall (Churn class): ~81%  

XGBoost provides the best balance for churn detection.

---

## Churn Probability System

The model generates churn probability for each customer.

Example:
Customer A → 6.6% churn probability  
Customer B → 60.7% churn probability  

High probability customers can be targeted with:
- Discounts
- Loyalty benefits
- Personalized offers
- Customer support calls

---

## Deployment

The final model was saved using joblib and deployed with Streamlit.

Saved files:
- final_churn_model.pkl
- threshold.pkl

The Streamlit web app allows users to:
- Input customer details
- Adjust charges and tenure using sliders
- Generate churn prediction instantly
- View churn probability score
- Visualize prediction with interactive UI

---

## Tech Stack

Programming Language:
- Python

Libraries:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

## Project Structure

project/
│
├── app.py
├── final_churn_model.pkl
├── threshold.pkl
├── model_test.py
├── requirements.txt
├── README.md
└── dataset.csv

---

## How to Run the Project

1. Install dependencies:

pip install -r requirements.txt

2. Run Streamlit app:

streamlit run app.py

3. Open browser:

http://localhost:8501

---

## Business Impact

This system helps companies:

- Detect high-risk customers early
- Reduce revenue loss
- Improve retention strategy
- Increase customer lifetime value
- Make data-driven decisions

The model focuses on high recall to ensure maximum churn customers are identified.

---

## Future Improvements

- SHAP explainability integration
- Feature importance dashboard
- Cloud deployment
- Automated retraining pipeline
- Customer segmentation analytics
- Admin analytics dashboard

---

## Conclusion

This project successfully implements a complete real-world churn prediction pipeline:

- Data cleaning and preprocessing
- EDA and business insights
- Model training and comparison
- Recall optimization
- Probability-based prediction
- Deployment with interactive UI

## Team Members
  Vanshika Yadav  
  Riya Garg  
  Ronit Singh  
  Sankalp  
