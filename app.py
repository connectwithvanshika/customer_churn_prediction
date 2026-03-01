import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------------------
# Streamlit Web Application
# Customer Churn Intelligence System
# This app takes customer profile input and predicts
# whether the customer is likely to churn or stay.
# ------------------------------------------------------------


# ── Page Configuration ─────────────────────────────────────
# Setting page title, icon and layout style

st.set_page_config(
    page_title="Customer Churn Intelligence System",
    page_icon="✦",
    layout="wide",
)


# ── Load Saved Model Artifacts ─────────────────────────────
# Using cache to avoid reloading model files on every interaction

@st.cache_resource
def load_model():
    return joblib.load("final_churn_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_threshold():
    return float(joblib.load("threshold.pkl"))

@st.cache_resource
def load_encoders():
    return joblib.load("encoders.pkl")


# Load all necessary components for prediction
model     = load_model()
scaler    = load_scaler()
threshold = load_threshold()
encoders  = load_encoders()


# ── Custom CSS Styling ─────────────────────────────────────
# Styling the entire UI for professional look and branding

st.markdown("""
<style>
...
</style>
""", unsafe_allow_html=True)


# ── Hero Section ───────────────────────────────────────────
# Main heading and introduction of the system

st.markdown("""
<div class="hero">
  <div class="badge">✦ ML-Powered Analytics</div>
  <div class="hero-title">
    Customer Churn<br>
    <span class="accent">Intelligence</span> System
  </div>
  <div class="hero-sub">
    Predict customer retention behavior using machine learning — fast,
    accurate & actionable.
  </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Customer Input Section ─────────────────────────────────
# Collecting all customer profile details required for prediction

st.markdown('<div class="section-label">Customer Profile Input</div>', unsafe_allow_html=True)

# Dividing input into structured columns for better UI layout
c1, c2, c3 = st.columns(3)

with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with c2:
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_s = st.selectbox("Online Security", ["No", "Yes"])
    online_b = st.selectbox("Online Backup", ["No", "Yes"])

with c3:
    device = st.selectbox("Device Protection", ["No", "Yes"])
    tech = st.selectbox("Tech Support", ["No", "Yes"])
    tv = st.selectbox("Streaming TV", ["No", "Yes"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

c4, c5 = st.columns(2)

with c4:
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with c5:
    monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, 0.5)
    total_c = st.slider("Total Charges ($)", 18.0, 9000.0, 1500.0, 50.0)


# ── Prediction Button ──────────────────────────────────────
# When user clicks this button, prediction process starts

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="predict-wrap">', unsafe_allow_html=True)
run = st.button("✦  Run Churn Prediction")
st.markdown("</div>", unsafe_allow_html=True)


# ── Prediction Logic ───────────────────────────────────────
# Executes only when button is clicked

if run:

    # Step 1: Encode categorical inputs using saved encoders
    enc = {
        "gender": encoders["gender"].transform([gender])[0],
        "SeniorCitizen": encoders["SeniorCitizen"].transform([senior])[0],
        "Partner": encoders["Partner"].transform([partner])[0],
        "Dependents": encoders["Dependents"].transform([dependents])[0],
        "tenure": tenure,
        "PhoneService": encoders["PhoneService"].transform([phone])[0],
        "MultipleLines": encoders["MultipleLines"].transform([multiple])[0],
        "InternetService": encoders["InternetService"].transform([internet])[0],
        "OnlineSecurity": encoders["OnlineSecurity"].transform([online_s])[0],
        "OnlineBackup": encoders["OnlineBackup"].transform([online_b])[0],
        "DeviceProtection": encoders["DeviceProtection"].transform([device])[0],
        "TechSupport": encoders["TechSupport"].transform([tech])[0],
        "StreamingTV": encoders["StreamingTV"].transform([tv])[0],
        "StreamingMovies": encoders["StreamingMovies"].transform([movies])[0],
        "Contract": encoders["Contract"].transform([contract])[0],
        "PaperlessBilling": encoders["PaperlessBilling"].transform([paperless])[0],
        "PaymentMethod": encoders["PaymentMethod"].transform([payment])[0],
        "MonthlyCharges": monthly,
        "TotalCharges": total_c,
    }

    # Step 2: Convert to DataFrame
    input_df = pd.DataFrame([enc])

    # Step 3: Ensure correct feature order (same as training)
    feature_order = joblib.load("feature_order.pkl")
    input_df = input_df[feature_order]

    # Step 4: Scale numerical columns
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Step 5: Get prediction probability
    prob = float(model.predict_proba(input_df)[0][1])

    # Step 6: Apply threshold to classify churn/stay
    will_churn = prob >= threshold

    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

    # Step 7: Display result in UI
    if will_churn:
        st.markdown("""
        <div class="result-card">
          <div class="result-icon"></div>
          <div class="result-title churn">Customer Likely to Churn</div>
          <div class="result-desc">
            High churn risk detected — immediate retention action recommended.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card">
          <div class="result-icon">✦</div>
          <div class="result-title">Customer Likely to Stay</div>
          <div class="result-desc">
            Low churn risk — customer shows strong loyalty signals.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Display probability score
    st.markdown(f"""
    <div class="prob-card">
      <div>
        <div class="prob-label">Churn Probability Score</div>
        <div class="prob-meta">Threshold: {threshold:.2f}</div>
      </div>
      <div class="prob-value">{prob * 100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Show progress bar based on churn probability
    st.progress(min(prob, 1.0))