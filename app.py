import streamlit as st
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Intelligence System",
    layout="wide"
)

# ================= LOAD MODEL =================
model = joblib.load("churn_xgboost_model.pkl")

# ================= TITLE =================
st.title("Customer Churn Intelligence System")
st.write("Predict customer retention behavior using XGBoost model")

st.markdown("---")

# ================= INPUT SECTION =================
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
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"]
    )

with col5:

    monthly = st.slider(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=500.0,
        value=70.0,
        step=1.0
    )

    total = st.slider(
        "Total Charges ($)",
        min_value=0.0,
        max_value=20000.0,
        value=1000.0,
        step=10.0
    )

# ================= PREDICTION =================
if st.button("Predict Customer Churn"):

    # ======== MANUAL ENCODING (IMPORTANT) ========

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

    # Internet encoding
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    internet = internet_map[internet]

    # Contract encoding
    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }
    contract = contract_map[contract]

    # Payment encoding
    payment_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }
    payment = payment_map[payment]

    # ======== Create DataFrame (ONLY NUMERIC) ========

    input_data = pd.DataFrame([[
        gender, senior, partner, dependents,
        tenure, phone, multiple, internet,
        online_security, online_backup,
        device, tech, tv, movies,
        contract, paperless, payment,
        monthly, total
    ]], columns=[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ])

    # ======== Prediction ========

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.subheader("Prediction: Customer is likely to churn")
    else:
        st.subheader("Prediction: Customer is likely to stay")

    st.write(f"Churn Probability: {round(prob*100,2)}%")
    st.progress(float(prob))