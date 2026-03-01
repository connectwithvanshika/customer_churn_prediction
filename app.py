import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence System",
    page_icon="✦",
    layout="wide",
)

# ── Load model + scaler + encoders ──────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("final_churn_model.pkl")
    except Exception:
        st.error("Error loading model file: 'final_churn_model.pkl'")
        st.stop()


@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler.pkl")
    except Exception:
        st.error("Error loading scaler file: 'scaler.pkl'")
        st.stop()


@st.cache_resource
def load_threshold():
    try:
        return float(joblib.load("threshold.pkl"))
    except Exception:
        st.error("Error loading threshold file: 'threshold.pkl'")
        st.stop()


@st.cache_resource
def load_encoders():
    try:
        return joblib.load("encoders.pkl")
    except Exception:
        st.error("Error loading encoders file: 'encoders.pkl'")
        st.stop()

@st.cache_resource
def load_feature_order():
    try:
        return joblib.load("feature_order.pkl")
    except Exception:
        st.error("Error loading feature_order.pkl")
        st.stop()

model     = load_model()
scaler    = load_scaler()
threshold = load_threshold()
encoders  = load_encoders()


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


# INPUT SECTION
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

    input_df = pd.DataFrame([enc])
    feature_order = load_feature_order()
    input_df = input_df[feature_order]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    with st.spinner("Running churn prediction..."):
      prob = float(model.predict_proba(input_df)[0][1])
    will_churn = prob >= threshold

    # Confidence score shows how strongly the model believes in prediction
    confidence = abs(prob - 0.5) * 2
    st.write(f"Model Confidence Score: {confidence:.2f}")

    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

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