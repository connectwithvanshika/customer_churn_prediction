"""
Customer Churn Intelligence System
-----------------------------------
This Streamlit application loads a trained XGBoost churn prediction model
and provides real-time churn probability predictions based on user inputs.

The pipeline ensures:
- Consistent feature encoding
- Correct feature ordering
- Proper numerical scaling
- Threshold-based classification
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np

import json
from langgraph.graph import StateGraph
from typing import TypedDict, List
from groq import Groq


# ── Page Configuration ─────────────────────────────────────────────
# Sets metadata and layout for the Streamlit application
st.set_page_config(
    page_title="Customer Churn Intelligence System",
    page_icon="✦",
    layout="wide",
)
@st.cache_resource
def load_model():
    """
    Loads the trained XGBoost model from disk.
    Stops the app if the model file is not found.
    """
    try:
        return joblib.load("notebook_&_otherpkl/final_churn_model.pkl")
    except Exception:
        st.error("Error loading model file: 'notebook_&_otherpkl/final_churn_model.pkl'")
        st.stop()


@st.cache_resource
def load_scaler():
    """
    Loads the StandardScaler used during training.
    Ensures consistent scaling between training and inference.
    """
    try:
        return joblib.load("notebook_&_otherpkl/scaler.pkl")
    except Exception:
        st.error("Error loading scaler file: 'notebook_&_otherpkl/scaler.pkl'")
        st.stop()

@st.cache_resource
def load_threshold():
    """
    Loads the optimized classification threshold.
    Threshold (0.4) was chosen to prioritize recall over accuracy.
    """
    try:
        return float(joblib.load("notebook_&_otherpkl/threshold.pkl"))
    except Exception:
        st.error("Error loading threshold file: 'notebook_&_otherpkl/threshold.pkl'")
        st.stop()


@st.cache_resource
def load_encoders():
    """
    Loads saved LabelEncoders for categorical features.
    Ensures consistent mapping between training and deployment.
    """
    try:
        return joblib.load("notebook_&_otherpkl/encoders.pkl")
    except Exception:
        st.error("Error loading encoders file: 'notebook_&_otherpkl/encoders.pkl'")
        st.stop()

@st.cache_resource
def load_feature_order():
    """
    Loads feature ordering used during model training.
    Prevents feature misalignment during prediction.
    """
    try:
        return joblib.load("notebook_&_otherpkl/feature_order.pkl")
    except Exception:
        st.error("Error loading feature_order.pkl")
        st.stop()


# Load all required artifacts
model     = load_model()
scaler    = load_scaler()
threshold = load_threshold()
encoders  = load_encoders()


# Load RAG knowledge base
with open("retention_knowledge.json") as f:
    knowledge = json.load(f)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class AgentState(TypedDict):
    churn_prob: float
    tenure: int
    monthly: float
    
    risk_level: str
    reasons: List[str]
    
    strategies: List[str]
    sources: List[str]
    
    final_output: str


def risk_node(state: AgentState):
    prob = state["churn_prob"]

    if prob > 0.7:
        risk = "High"
    elif prob > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    reasons = []

    if state["tenure"] < 6:
        reasons.append("low_tenure")

    if state["monthly"] > 80:
        reasons.append("high_charges")

    if not reasons:
        reasons.append("general")

    return {**state, "risk_level": risk, "reasons": reasons}


def retrieval_node(state: AgentState):
    strategies = []
    sources = []

    for item in knowledge:
        if item["condition"] in state["reasons"]:
            strategies.append(item["strategy"])
            sources.append(item["source"])

    return {
        **state,
        "strategies": list(set(strategies)),
        "sources": list(set(sources))
    }

# ── CSS (UNCHANGED) ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,700;1,700&display=swap');

html, body, [data-testid="stAppViewContainer"], section.main {
    background: #ede9ff !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container {
    max-width: 1400px !important;
    padding: 2rem 3rem 4rem !important;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: white;
    border: 1px solid #d4c6ff;
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6d28d9;
    margin-bottom: 1.4rem;
}

.hero {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 700;
    line-height: 1.1;
    color: #1a0a3d;
    margin-bottom: 1rem;
}
.hero-title .accent {
    font-style: italic;
    background: linear-gradient(90deg, #7c3aed, #c026d3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.92rem;
    color: #6b7280;
    line-height: 1.7;
    max-width: 420px;
    margin: 0 auto;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c4b5fd, transparent);
    margin: 1.8rem 0;
}

.section-label {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 1.1rem;
}
.section-label::before {
    content: '';
    width: 4px;
    height: 14px;
    background: #7c3aed;
    border-radius: 2px;
    flex-shrink: 0;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #d4c6ff, transparent);
}

[data-testid="stHorizontalBlock"] > div {
    background: white !important;
    border: 1px solid #e9e3ff !important;
    border-radius: 16px !important;
    padding: 1.4rem 1.3rem !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.06) !important;
}

[data-testid="stWidgetLabel"] p {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #9ca3af !important;
    margin-bottom: 4px !important;
}

[data-testid="stSelectbox"] > div > div {
    background: white !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
}

[data-testid="stSlider"] [role="slider"] {
    background: #7c3aed !important;
    border: none !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
}
[data-testid="stSlider"] p {
    color: #7c3aed !important;
    font-weight: 600 !important;
}

.predict-wrap .stButton > button {
    width: 100% !important;
    padding: 1.05rem !important;
    background: linear-gradient(90deg, #7c3aed, #c026d3) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
}
.predict-wrap .stButton > button:hover {
    opacity: 0.9 !important;
}

.result-card {
    background: linear-gradient(135deg, #ede9ff 0%, #fae8ff 100%);
    border: 1px solid #d4c6ff;
    border-radius: 18px;
    padding: 2.6rem 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-icon { font-size: 1.8rem; margin-bottom: 0.8rem; }
.result-title {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7c3aed, #c026d3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.6rem;
}
.result-title.churn {
    background: linear-gradient(90deg, #dc2626, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.result-desc {
    font-size: 0.85rem;
    color: #6b7280;
    line-height: 1.7;
    max-width: 380px;
    margin: 0 auto;
}

.prob-card {
    background: white;
    border: 1px solid #e9e3ff;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}
.prob-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #9ca3af;
}
.prob-meta { font-size: 0.75rem; color: #9ca3af; }
.prob-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7c3aed, #c026d3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

[data-testid="stProgress"] > div > div {
    background: #e9e3ff !important;
    border-radius: 99px !important;
    height: 7px !important;
}
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #c026d3) !important;
    border-radius: 99px !important;
}
</style>
""", unsafe_allow_html=True)


# HERO
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


# ── User Input Section ─────────────────────────────────────────────
# Collects structured customer profile information
st.markdown('<div class="section-label">Customer Profile Input</div>', unsafe_allow_html=True)

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

# ── Prediction Trigger ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="predict-wrap">', unsafe_allow_html=True)
run = st.button("✦  Run Churn Prediction")
st.markdown("</div>", unsafe_allow_html=True)


if run:
    # Input validation to prevent invalid numeric values
    if monthly < 0 or total_c < 0:
        st.warning("Charges cannot be negative.")
        st.stop()

    if tenure < 0:
        st.warning("Tenure cannot be negative.")
        st.stop()

    
    # Encode categorical variables using saved encoders
    # Ensures deployment uses same encoding scheme as training

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


    # Convert input dictionary into DataFrame
    input_df = pd.DataFrame([enc])

    # Ensure feature ordering matches training pipeline
    feature_order = load_feature_order()
    input_df = input_df[feature_order]

    # Scale numerical columns using saved StandardScaler
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])



    # Predict churn probability
    with st.spinner("Running churn prediction..."):
      prob = float(model.predict_proba(input_df)[0][1])

    # Apply optimized threshold (0.4) instead of default 0.5
    will_churn = prob >= threshold

    # Confidence score shows how strongly the model believes in prediction
    st.progress(min(prob, 1.0))

    # Confidence score indicates distance from decision boundary (0.5)
    confidence = abs(prob - 0.5) * 2
    st.caption(f"Model Confidence Score: {confidence:.2f}")

    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)


    # Classification Output
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

    st.markdown(f"""
    <div class="prob-card">
      <div>
        <div class="prob-label">Churn Probability Score</div>
        <div class="prob-meta">Threshold: {threshold:.2f}</div>
      </div>
      <div class="prob-value">{prob * 100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(prob, 1.0))