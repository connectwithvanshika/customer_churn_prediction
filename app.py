import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- PREMIUM LIGHT UI ----------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=Inter:wght@300;400&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    background-color: #ffffff;
    color: #000000;
    font-family: 'Inter', sans-serif;
}

/* MAIN TITLE */
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    text-align: center;
    margin-top: 40px;
    margin-bottom: 10px;
    letter-spacing: 1px;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 50px;
}

/* SECTION TITLE */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    margin-top: 40px;
    margin-bottom: 20px;
}

/* INPUT BOX */
.stSelectbox, .stNumberInput {
    background-color: #fafafa;
}

/* BUTTON */
div.stButton > button {
    background-color: black;
    color: white;
    padding: 14px 32px;
    font-size: 16px;
    border-radius: 0px;
    border: 1px solid black;
    transition: 0.3s;
    display: block;
    margin: 40px auto;
}

/* HOVER */
div.stButton > button:hover {
    background-color: white;
    color: black;
    border: 1px solid black;
}

/* CLICK FIX 🔥 */
div.stButton > button:focus,
div.stButton > button:active {
    background-color: black !important;
    color: white !important;
    outline: none !important;
    box-shadow: none !important;
}

/* AFTER CLICK HOLD */
div.stButton > button:focus:not(:active) {
    background-color: black !important;
    color: white !important;
}


/* RESULT BOX */
.result-box {
    text-align:center;
    padding:30px;
    border:1px solid #eee;
    margin-top:40px;
    font-size:22px;
    font-family: 'Playfair Display', serif;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown('<div class="main-title">Customer Churn Prediction Model</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict customer retention using machine learning insights</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="section-title">Customer Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male","Female"])
    senior = st.selectbox("Senior Citizen", ["Yes","No"])
    partner = st.selectbox("Partner", ["Yes","No"])
    dependents = st.selectbox("Dependents", ["Yes","No"])
    tenure = st.number_input("Tenure (months)",0,120)

with col2:
    phone = st.selectbox("Phone Service", ["Yes","No"])
    multiple = st.selectbox("Multiple Lines", ["Yes","No"])
    internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
    online_security = st.selectbox("Online Security", ["Yes","No"])
    online_backup = st.selectbox("Online Backup", ["Yes","No"])

with col3:
    device = st.selectbox("Device Protection", ["Yes","No"])
    tech = st.selectbox("Tech Support", ["Yes","No"])
    tv = st.selectbox("Streaming TV", ["Yes","No"])
    movies = st.selectbox("Streaming Movies", ["Yes","No"])
    contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])

col4, col5, col6 = st.columns(3)

with col4:
    paperless = st.selectbox("Paperless Billing", ["Yes","No"])

with col5:
    payment = st.selectbox("Payment Method",
    ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

with col6:
    monthly = st.number_input("Monthly Charges",0.0,500.0)
    total = st.number_input("Total Charges",0.0,20000.0)

# ---------------- ENCODING ----------------
def encode():

    gender_val = 1 if gender=="Male" else 0
    senior_val = 1 if senior=="Yes" else 0
    partner_val = 1 if partner=="Yes" else 0
    dependents_val = 1 if dependents=="Yes" else 0
    phone_val = 1 if phone=="Yes" else 0
    multiple_val = 1 if multiple=="Yes" else 0
    paperless_val = 1 if paperless=="Yes" else 0

    # internet
    if internet=="DSL": internet_val=0
    elif internet=="Fiber optic": internet_val=1
    else: internet_val=2

    # contract
    if contract=="Month-to-month": contract_val=0
    elif contract=="One year": contract_val=1
    else: contract_val=2

    # payment
    pay_map={
        "Electronic check":0,
        "Mailed check":1,
        "Bank transfer (automatic)":2,
        "Credit card (automatic)":3
    }
    payment_val=pay_map[payment]

    # services
    online_sec_val = 1 if online_security=="Yes" else 0
    online_backup_val = 1 if online_backup=="Yes" else 0
    device_val = 1 if device=="Yes" else 0
    tech_val = 1 if tech=="Yes" else 0
    tv_val = 1 if tv=="Yes" else 0
    movies_val = 1 if movies=="Yes" else 0

    data = np.array([[gender_val, senior_val, partner_val, dependents_val, tenure,
                      phone_val, multiple_val, internet_val,
                      online_sec_val, online_backup_val, device_val, tech_val,
                      tv_val, movies_val, contract_val, paperless_val,
                      payment_val, monthly, total]])

    data = scaler.transform(data)
    return data

# ---------------- PREDICTION ----------------
if st.button("Predict Customer Behavior"):

    final = encode()
    pred = model.predict(final)[0]

    if pred==1:
        st.markdown('<div class="result-box">Customer is likely to churn</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box">Customer is likely to stay</div>', unsafe_allow_html=True)
