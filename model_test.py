# Local testing of model 

import joblib
import pandas as pd

# load model
model = joblib.load("churn_xgboost_model.pkl")

print("Model loaded successfully")

# sample test data (same format as training)
test_data = pd.DataFrame([[
    1,0,1,0,12,1,0,1,0,1,0,0,1,0,0,1,2,70.0,1000.0
]], columns=[
    'gender','SeniorCitizen','Partner','Dependents',
    'tenure','PhoneService','MultipleLines','InternetService',
    'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
    'StreamingTV','StreamingMovies','Contract','PaperlessBilling',
    'PaymentMethod','MonthlyCharges','TotalCharges'
])

pred = model.predict(test_data)[0]
prob = model.predict_proba(test_data)[0][1]

print("Prediction:", pred)
print("Churn probability:", round(prob*100,2), "%")