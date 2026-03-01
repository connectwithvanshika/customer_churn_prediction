import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------------------
# This file is used to test the trained churn model separately
# from the Streamlit UI. It helps verify that:
# 1. The model loads correctly
# 2. Encoders & scaler work properly
# 3. Feature order matches training
# 4. Prediction + probability calculation works
# ------------------------------------------------------------


# ── Load all saved model artifacts ────────────────────────
# These were saved after training the model

model = joblib.load("final_churn_model.pkl")      # Trained ML model
scaler = joblib.load("scaler.pkl")                # Scaler used for numeric features
threshold = float(joblib.load("threshold.pkl"))   # Custom decision threshold
encoders = joblib.load("encoders.pkl")            # Label encoders for categorical features
feature_order = joblib.load("feature_order.pkl")  # Exact feature order used during training

print("All model artifacts loaded successfully.\n")


# ── Create a sample test customer manually ────────────────
# This simulates real customer input for testing purposes

sample_customer = {
    "gender": "Male",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 2100.0
}


# ── Encode categorical features ────────────────────────────
# Convert text values into numeric format using saved encoders

encoded = {}

for col in sample_customer:
    if col in encoders:
        # Apply the same encoder used during training
        encoded[col] = encoders[col].transform([sample_customer[col]])[0]
    else:
        # Numeric columns remain unchanged
        encoded[col] = sample_customer[col]


# ── Convert dictionary to DataFrame ───────────────────────
# Model expects input in DataFrame format

input_df = pd.DataFrame([encoded])


# ── Ensure correct feature order ──────────────────────────
# Very important: model expects columns in same order as training

input_df = input_df[feature_order]


# ── Scale numeric columns ─────────────────────────────────
# Apply the same scaler used during model training

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
input_df[num_cols] = scaler.transform(input_df[num_cols])


# ── Make prediction ───────────────────────────────────────
# Get probability of churn (class 1)

prob = float(model.predict_proba(input_df)[0][1])

# Apply threshold to determine final decision
prediction = prob >= threshold


# ── Display results ───────────────────────────────────────

print("Prediction Results")
print("----------------------------------")
print(f"Churn Probability: {prob:.4f}")
print(f"Threshold Used: {threshold}")
print(f"Final Prediction: {'Customer WILL churn' if prediction else 'Customer will NOT churn'}")