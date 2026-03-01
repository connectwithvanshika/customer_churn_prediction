# Project 5: Customer Churn Prediction & Agentic Retention Strategy

## From Predictive Analytics to Intelligent Intervention

---

## 1. Project Overview

This project presents the design and implementation of an AI-driven customer analytics system that predicts customer churn and evolves into an agentic AI-powered retention strategist.

The system is divided into two milestones:

### Milestone 1: ML-Based Churn Prediction

A classical machine learning pipeline was developed to predict whether a telecom customer is likely to churn based on:

- Service usage
- Contract type
- Billing details
- Demographic attributes

The goal is to identify high-risk customers early so that proactive retention strategies can be applied.

### Milestone 2: Agentic Retention Strategist (Extension)

The system will be extended into an intelligent agent that:

- Reasons about churn probability
- Retrieves retention best practices using RAG (Retrieval Augmented Generation)
- Generates structured intervention strategies
- Plans retention workflows autonomously

---

## 2.Problem Statement

Customer churn is a critical issue for subscription-based businesses because losing existing customers directly impacts revenue and long-term growth. Acquiring new customers is significantly more expensive than retaining current ones.

The objective of this project is to:

- Predict customers who are likely to churn
- Identify key factors contributing to churn
- Support proactive and data-driven retention strategies
- Transition from predictive analytics to intelligent AI-assisted intervention

---

## 3. Dataset Information

Dataset: Telco Customer Churn Dataset  
Total Records: 7043  
Features: 20 input variables + 1 target variable  

Target Variable:
- `Churn` (Yes/No)

Feature Categories:

1. Customer Demographics
   - Gender
   - SeniorCitizen
   - Partner
   - Dependents

2. Services Subscribed
   - PhoneService
   - MultipleLines
   - InternetService
   - OnlineSecurity
   - OnlineBackup
   - DeviceProtection
   - TechSupport
   - StreamingTV
   - StreamingMovies

3. Account Information
   - Tenure
   - Contract
   - PaperlessBilling
   - PaymentMethod
   - MonthlyCharges
   - TotalCharges

---

## 4. Data Preprocessing

The following preprocessing steps were performed:

- Removed irrelevant column: `customerID`
- Converted `TotalCharges` to numeric and handled missing values
- Dropped 11 rows with zero tenure
- Label encoded categorical features
- Standardized numerical features:
  - tenure
  - MonthlyCharges
  - TotalCharges
- Stratified train-test split to handle class imbalance
- Saved preprocessing artifacts:
  - encoders.pkl
  - scaler.pkl
  - feature_order.pkl

---

## 5. Model Development

Three models were trained and evaluated:

- Logistic Regression
- Decision Tree
- XGBoost (Final Selected Model)

### Why XGBoost?

Although Logistic Regression showed slightly higher accuracy, XGBoost achieved the highest recall for churn customers.

Recall is prioritized in churn prediction because missing a potential churn customer is more costly than incorrectly flagging a loyal one.

Key Metrics (XGBoost):
- High recall for churn class
- Balanced precision-recall tradeoff
- Improved performance after threshold tuning

---

## 6. Threshold Optimization

Default classification threshold (0.5) was adjusted to improve recall.

Instead of:

```
prediction = model.predict(X_test)
```

The system uses:

```
probabilities = model.predict_proba(X_test)[:,1]
prediction = (probabilities > 0.4).astype(int)
```

This improves churn detection performance.

Threshold value is saved in:
```
threshold.pkl
```

---

## 7. Key Churn Drivers Identified

EDA revealed strong predictors of churn:

- Month-to-month contracts increase churn
- Electronic check payment method correlates with higher churn
- Low tenure customers churn more frequently
- Lack of tech support increases churn probability
- High monthly charges increase churn risk
- Customers without online security or backup services churn more

Gender showed minimal impact on churn.

---

## 8. Streamlit Deployment

A production-ready Streamlit application was developed.

Features:

- Interactive customer profile input
- Real-time churn probability prediction
- Custom threshold-based classification
- Styled UI with responsive layout
- Probability visualization with progress bar
- Retention recommendation messaging

The application correctly:

- Applies saved encoders
- Reorders features to match training
- Scales numerical columns
- Uses the saved model and threshold

---

## 9. Project Architecture

### Milestone 1 Architecture

User Input → Encoding → Feature Ordering → Scaling → XGBoost Model → Probability → Threshold Logic → UI Output

### Milestone 2 (Planned Agent Architecture)

User Query → Risk Assessment → RAG Retrieval → Strategy Planning → Structured Retention Report

---

## 10. Project Structure

```
customer_churn_prediction/
│
├── app.py
├── model_test.py
├── requirements.txt
├── README.md
│
├── final_churn_model.pkl
├── scaler.pkl
├── encoders.pkl
├── threshold.pkl
├── feature_order.pkl
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── notebooks/
    └── CUSTOMER_CHURN_PREDICTION.ipynb
```

---

## 11. Technology Stack

| Component | Technology |
|------------|------------|
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML Models | Scikit-learn, XGBoost |
| Preprocessing | StandardScaler, LabelEncoder |
| Deployment | Streamlit |
| Model Storage | Joblib |

Planned for Milestone 2:
- LangGraph
- Chroma or FAISS
- Open-source LLM (Free tier)
- Hosted deployment (Hugging Face / Streamlit Cloud / Render)

---

## 12. Milestone Deliverables

### Milestone 1 (Completed)

- Business understanding
- EDA and feature engineering
- Model comparison
- Threshold optimization
- Working local Streamlit application
- Model artifacts saved
- Performance evaluation report

### Milestone 2 (Planned)

- Agentic retention strategist
- RAG-based best practice retrieval
- Structured retention reports
- Public deployment link
- Agent workflow documentation
- GitHub repository
- Demo video

---

## 13. Future Improvements

- Feature importance visualization in UI
- SHAP explanations for transparency
- Dynamic threshold tuning
- Customer segmentation module
- Agentic reasoning layer using LangGraph
- Automated retention playbook generation

---

## 14. Conclusion

This project successfully implements a complete churn prediction pipeline from raw dataset to deployed interactive application.

It transitions from classical machine learning to the foundation of an agent-based intelligent retention strategist.

The system:

- Identifies high-risk customers
- Provides probability-based insights
- Enables proactive intervention
- Is deployment-ready
- Is extensible toward autonomous AI decision systems

This demonstrates practical application of machine learning in business analytics and lays the foundation for intelligent AI-driven customer retention systems.

Developed by Team RetainX AI

Team Members 
1. Vanshika Yadav 
2. ⁠Riya Garg 
3. ⁠Sankalp 
4. ⁠Ronit Singh
