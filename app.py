import streamlit as st
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Intelligence System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= LOAD MODEL =================
model = joblib.load("final_churn_model.pkl")

# ================= PREMIUM CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700;1,900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* ═══════════════════════════════════════
   DESIGN TOKENS  (pink-purple system)
   --pp-1  deep violet    #6d28d9
   --pp-2  purple         #9333ea
   --pp-3  violet-mid     #a855f7
   --pp-4  lavender       #c084fc
   --pp-5  pink           #e879f9  /  #ec4899
   --pp-bg soft cream     #faf8ff
═══════════════════════════════════════ */
:root {
    --pp-1: #6d28d9;
    --pp-2: #9333ea;
    --pp-3: #a855f7;
    --pp-4: #c084fc;
    --pp-5: #e879f9;
    --pp-pink: #ec4899;
    --grad: linear-gradient(135deg, #9333ea 0%, #c084fc 50%, #ec4899 100%);
    --grad-h: linear-gradient(135deg, #7e22ce 0%, #a855f7 50%, #db2777 100%);
    --surface: #ffffff;
    --bg: #faf8ff;
    --border: #ede8fc;
    --border-h: #c084fc;
    --text-main: #1a0a2e;
    --text-muted: #7c6f94;
    --text-soft: #a89dbf;
}

/* ═══════════════════════════════════════ GLOBAL */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--bg) !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-main);
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    max-width: 1260px !important;
    padding: 0 2.5rem 5rem !important;
}

/* ═══════════════════════════════════════ ANIMATIONS */
@keyframes slideDown  { from{opacity:0;transform:translateY(-44px)} to{opacity:1;transform:translateY(0)} }
@keyframes slideUp    { from{opacity:0;transform:translateY(44px)}  to{opacity:1;transform:translateY(0)} }
@keyframes slideRight { from{opacity:0;transform:translateX(-54px)} to{opacity:1;transform:translateX(0)} }
@keyframes slideLeft  { from{opacity:0;transform:translateX(54px)}  to{opacity:1;transform:translateX(0)} }
@keyframes popIn      { from{opacity:0;transform:scale(0.75)}       to{opacity:1;transform:scale(1)} }
@keyframes barFill    { from{width:0%} }
@keyframes pulseDot {
    0%,100% { box-shadow: 0 0 0 0 rgba(168,85,247,0.6); }
    50%     { box-shadow: 0 0 0 7px rgba(168,85,247,0); }
}
@keyframes shimmer {
    0%   { background-position: -600px 0; }
    100% { background-position: 600px 0; }
}
@keyframes gradMove {
    0%,100% { background-position: 0% 50%; }
    50%     { background-position: 100% 50%; }
}

/* ═══════════════════════════════════════ HERO */
.hero-wrap {
    padding: 4rem 0 2.8rem;
    text-align: center;
    animation: slideDown 0.75s cubic-bezier(0.22,1,0.36,1) both;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #fff;
    border: 1.5px solid #e9d5ff;
    border-radius: 999px;
    padding: 0.44rem 1.2rem;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--pp-2);
    margin-bottom: 1.6rem;
    box-shadow: 0 2px 16px rgba(147,51,234,0.13);
}

.hero-badge-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--grad);
    animation: pulseDot 2s infinite;
    flex-shrink: 0;
}

.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(2.8rem, 5.5vw, 4.4rem);
    font-weight: 900;
    line-height: 1.08;
    color: var(--text-main);
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}

/* "Intelligence" word — pink-purple gradient */
.hero-title em {
    font-style: italic;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 400;
    max-width: 490px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ═══════════════════════════════════════ DIVIDER */
.fancy-divider {
    height: 1.5px;
    background: linear-gradient(90deg, transparent 0%, #c084fc 35%, #ec4899 65%, transparent 100%);
    margin: 2rem 0;
    border-radius: 99px;
}

/* ═══════════════════════════════════════ SECTION LABEL */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.6rem;
    animation: slideRight 0.5s cubic-bezier(0.22,1,0.36,1) 0.1s both;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1.5px;
    background: linear-gradient(90deg, #e9d5ff, transparent);
    border-radius: 99px;
    -webkit-text-fill-color: unset;
}

/* ═══════════════════════════════════════ INPUT COLUMN CARDS */
[data-testid="stHorizontalBlock"] > div:nth-child(1) {
    animation: slideRight 0.6s cubic-bezier(0.22,1,0.36,1) 0.05s both;
}
[data-testid="stHorizontalBlock"] > div:nth-child(2) {
    animation: slideUp 0.6s cubic-bezier(0.22,1,0.36,1) 0.15s both;
}
[data-testid="stHorizontalBlock"] > div:nth-child(3) {
    animation: slideLeft 0.6s cubic-bezier(0.22,1,0.36,1) 0.25s both;
}

[data-testid="stHorizontalBlock"] > div {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 22px;
    padding: 1.7rem 1.5rem !important;
    box-shadow: 0 4px 26px rgba(147,51,234,0.07);
    transition: box-shadow 0.3s ease, border-color 0.3s ease, transform 0.3s ease;
}
[data-testid="stHorizontalBlock"] > div:hover {
    box-shadow: 0 10px 42px rgba(147,51,234,0.16);
    border-color: var(--border-h);
    transform: translateY(-3px);
}

/* ═══════════════════════════════════════ WIDGET LABELS */
[data-testid="stWidgetLabel"] p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-soft) !important;
}

/* ═══════════════════════════════════════ SELECTBOX */
[data-testid="stSelectbox"] > div > div {
    background: #fdf8ff !important;
    border: 1.5px solid #ede8fc !important;
    border-radius: 12px !important;
    color: var(--text-main) !important;
    transition: border-color 0.22s, box-shadow 0.22s !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--pp-4) !important;
    box-shadow: 0 0 0 3px rgba(192,132,252,0.18) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--pp-2) !important;
    box-shadow: 0 0 0 3px rgba(147,51,234,0.14) !important;
}
[data-testid="stSelectbox"] ul {
    background: #fff !important;
    border: 1.5px solid #ede8fc !important;
    border-radius: 14px !important;
    box-shadow: 0 14px 44px rgba(147,51,234,0.16) !important;
}
[data-testid="stSelectbox"] li {
    color: #374151 !important;
    border-radius: 8px !important;
    font-size: 0.87rem !important;
}
[data-testid="stSelectbox"] li:hover,
[data-testid="stSelectbox"] li[aria-selected="true"] {
    background: #fdf4ff !important;
    color: var(--pp-2) !important;
}

/* ═══════════════════════════════════════ SLIDER */
[data-testid="stSlider"] > div > div > div {
    background: #f3e8ff !important;
    border-radius: 99px !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--pp-2), var(--pp-pink)) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #fff !important;
    border: 3px solid var(--pp-2) !important;
    box-shadow: 0 2px 14px rgba(147,51,234,0.38) !important;
    width: 20px !important; height: 20px !important;
}
[data-testid="stSlider"] p {
    color: var(--pp-2) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* ═══════════════════════════════════════ PREDICT BUTTON */
.stButton > button {
    width: 100% !important;
    padding: 1.05rem 2rem !important;
    background: var(--grad) !important;
    background-size: 200% 200% !important;
    animation: gradMove 4s ease infinite !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    box-shadow: 0 6px 28px rgba(147,51,234,0.38) !important;
    transition: transform 0.18s, box-shadow 0.2s !important;
    margin-top: 1.2rem !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.22) 50%, transparent 70%);
    background-size: 600px 100%;
    animation: shimmer 2.8s infinite linear;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 36px rgba(147,51,234,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ═══════════════════════════════════════ RESULT CARDS */
.result-churn {
    background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    border: 2px solid #fca5a5;
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    animation: slideUp 0.6s cubic-bezier(0.22,1,0.36,1) both;
    position: relative; overflow: hidden;
}
.result-churn::before {
    content: '';
    position: absolute; top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(239,68,68,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.result-stay {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 2px solid #c084fc;
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    animation: slideUp 0.6s cubic-bezier(0.22,1,0.36,1) both;
    position: relative; overflow: hidden;
}
.result-stay::before {
    content: '';
    position: absolute; top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(168,85,247,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.result-icon {
    font-size: 3.6rem;
    display: block;
    margin-bottom: 0.8rem;
    animation: popIn 0.5s cubic-bezier(0.22,1,0.36,1) 0.1s both;
}

.result-title-churn {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2rem; font-weight: 900;
    color: #dc2626;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
}
.result-title-stay {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2rem; font-weight: 900;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
}
.result-sub {
    font-size: 0.88rem;
    color: var(--text-muted);
    font-weight: 400;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ═══════════════════════════════════════ PROBABILITY CARD */
.prob-wrap {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-top: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 24px rgba(147,51,234,0.08);
    animation: slideUp 0.6s cubic-bezier(0.22,1,0.36,1) 0.12s both;
}
.prob-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-soft);
    margin-bottom: 0.3rem;
}
.prob-desc { font-size: 0.84rem; color: var(--text-soft); font-weight: 400; }
.prob-value {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3rem; font-weight: 900;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    animation: popIn 0.5s cubic-bezier(0.22,1,0.36,1) 0.2s both;
}

/* ═══════════════════════════════════════ PROGRESS BAR */
[data-testid="stProgress"] {
    animation: slideUp 0.6s cubic-bezier(0.22,1,0.36,1) 0.22s both;
}
[data-testid="stProgress"] > div > div {
    background: #f3e8ff !important;
    border-radius: 99px !important;
    height: 12px !important;
}
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #9333ea 0%, #c084fc 50%, #ec4899 100%) !important;
    border-radius: 99px !important;
    box-shadow: 0 0 12px rgba(168,85,247,0.4) !important;
    animation: barFill 1.2s cubic-bezier(0.22,1,0.36,1) both !important;
}

/* ═══════════════════════════════════════ MISC */
[data-testid="stHorizontalBlock"] { gap: 1.4rem !important; }
hr { border-color: #ede8fc !important; margin: 2rem 0 !important; }
.stMarkdown p { color: var(--text-muted) !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #d8b4fe; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--pp-4); }
</style>
""", unsafe_allow_html=True)

# ================= HERO HEADER =================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">
        <div class="hero-badge-dot"></div>
        ML-Powered Analytics
    </div>
    <div class="hero-title">Customer Churn<br><em>Intelligence</em> System</div>
    <div class="hero-sub">Predict customer retention behavior using machine learning — fast, accurate &amp; actionable.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">🧑‍💼 &nbsp;Customer Profile Input</div>', unsafe_allow_html=True)

# ================= INPUT UI =================
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])

with col3:
    device = st.selectbox("Device Protection", ["No", "Yes"])
    tech = st.selectbox("Tech Support", ["No", "Yes"])
    tv = st.selectbox("Streaming TV", ["No", "Yes"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

col4, col5 = st.columns(2)

with col4:
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

with col5:
    monthly = st.slider(
        "Monthly Charges ($)",
        min_value=18.0, max_value=120.0, value=70.0, step=1.0
    )
    total = st.slider(
        "Total Charges ($)",
        min_value=18.0, max_value=9000.0, value=1500.0, step=50.0
    )

st.markdown("")

# ================= PREDICTION =================
if st.button("✦  Run Churn Prediction"):
    # ===== Encoding =====
    gender = 1 if gender == "Male" else 0
    senior = 1 if senior == "Yes" else 0
    partner = 1 if partner == "Yes" else 0
    dependents = 1 if dependents == "Yes" else 0
    phone = 1 if phone == "Yes" else 0
    multiple = 1 if multiple == "Yes" else 0
    online_security = 1 if online_security == "Yes" else 0
    online_backup = 1 if online_backup == "Yes" else 0
    device = 1 if device == "Yes" else 0
    tech = 1 if tech == "Yes" else 0
    tv = 1 if tv == "Yes" else 0
    movies = 1 if movies == "Yes" else 0
    paperless = 1 if paperless == "Yes" else 0
    internet = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet]
    contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
    payment = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }[payment]

    input_data = pd.DataFrame([[
        gender, senior, partner, dependents, tenure,
        phone, multiple, internet, online_security, online_backup,
        device, tech, tv, movies, contract, paperless,
        payment, monthly, total
    ]], columns=[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ])

    # ================= MODEL PREDICTION =================
    prob = model.predict_proba(input_data)[0][1]

    # ⭐ CUSTOM THRESHOLD (IMPORTANT)
    threshold = 0.4  # same used during training for 81% recall

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 &nbsp;Prediction Result</div>', unsafe_allow_html=True)

    # ================= RESULT UI =================
    if prob > threshold:
        st.markdown("""
        <div class="result-churn">
            <span class="result-icon">⚠️</span>
            <div class="result-title-churn">Customer Likely to Churn</div>
            <div class="result-sub">High risk of customer leaving — consider launching a targeted retention campaign immediately.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-stay">
            <span class="result-icon">✦</span>
            <div class="result-title-stay">Customer Likely to Stay</div>
            <div class="result-sub">Low churn risk — this customer shows strong loyalty signals. Continue nurturing the relationship.</div>
        </div>
        """, unsafe_allow_html=True)

    # Probability card
    st.markdown(f"""
    <div class="prob-wrap">
        <div>
            <div class="prob-label">Churn Probability Score</div>
            <div class="prob-desc">Threshold: 0.40 &nbsp;·&nbsp; Recall-optimised model</div>
        </div>
        <div class="prob-value">{prob*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Animated progress bar
    st.progress(float(prob))