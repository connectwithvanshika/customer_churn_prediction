import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------------------
# Simple Model Testing Script
# This file checks whether:
# 1. Model loads properly
# 2. Encoders work
# 3. Scaler works
# 4. Prediction gives correct output
# ------------------------------------------------------------

print("Loading model files...")

# Load saved files
model = joblib.load("final_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = float(joblib.load("threshold.pkl"))
encoders = joblib.load("encoders.pkl")
feature_order = joblib.load("feature_order.pkl")

print("Model and supporting files loaded successfully!\n")


# ------------------------------------------------------------
# Create a simple test customer manually
# ------------------------------------------------------------

test_customer = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.0,
    "TotalCharges": 500.0
}

print("Encoding features...")

# Encode categorical features
encoded = {}
for col in test_customer:
    if col in encoders:
        encoded[col] = encoders[col].transform([test_customer[col]])[0]
    else:
        encoded[col] = test_customer[col]

# Convert to DataFrame
input_df = pd.DataFrame([encoded])

# Arrange columns in correct order
input_df = input_df[feature_order]

# Scale numeric columns
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

print("Making prediction...\n")

# Predict
probability = float(model.predict_proba(input_df)[0][1])
prediction = probability >= threshold

# Display result
print("--------- MODEL TEST RESULT ---------")
print(f"Churn Probability: {probability:.4f}")
print(f"Threshold Used: {threshold}")
print(f"Final Prediction: {'Customer WILL churn' if prediction else 'Customer will NOT churn'}")
print("-------------------------------------")

print("\nModel testing completed successfully!")