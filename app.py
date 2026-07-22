"""
Customer Churn Predictor - Streamlit App
=========================================
Loads a RandomForestClassifier and a LogisticRegression model (both trained
on the Telco Customer Churn dataset), scales inputs with the matching
StandardScaler, and predicts churn probability from a friendly form.

Run with:
    streamlit run app.py

Make sure these 4 files are in the same folder as this script:
    random_forest_model.pkl
    logistic_regression_model.pkl
    scaler.pkl
    features.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
LR_PATH = os.path.join(BASE_DIR, "logistic_regression_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")


@st.cache_resource
def load_artifacts():
    missing = [p for p in [RF_PATH, LR_PATH, SCALER_PATH, FEATURES_PATH] if not os.path.exists(p)]
    if missing:
        st.error(
            "Missing model file(s):\n"
            + "\n".join(f"- {os.path.basename(m)}" for m in missing)
            + "\n\nPlace all 4 .pkl files in the same folder as app.py."
        )
        st.stop()

    rf = joblib.load(RF_PATH)
    lr = joblib.load(LR_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return rf, lr, scaler, features


rf_model, lr_model, scaler, FEATURES = load_artifacts()

# ---------------------------------------------------------------------------
# Encoding maps (alphabetical LabelEncoder scheme used by the original
# Telco Customer Churn training notebook that produced these models)
# ---------------------------------------------------------------------------
BINARY_YES_NO = {"No": 0, "Yes": 1}
GENDER_MAP = {"Female": 0, "Male": 1}
THREE_WAY_SERVICE = {"No": 0, "No internet service": 1, "Yes": 2}
MULTIPLE_LINES_MAP = {"No": 0, "No phone service": 1, "Yes": 2}
INTERNET_SERVICE_MAP = {"DSL": 0, "Fiber optic": 1, "No": 2}
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT_METHOD_MAP = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.caption("RandomForest & Logistic Regression models trained on the Telco Customer Churn dataset")

model_choice = st.radio("Choose model", ["Random Forest", "Logistic Regression"], horizontal=True)

st.subheader("Customer details")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", list(MULTIPLE_LINES_MAP.keys()))
    internet_service = st.selectbox("Internet Service", list(INTERNET_SERVICE_MAP.keys()))
    online_security = st.selectbox("Online Security", list(THREE_WAY_SERVICE.keys()))

with col2:
    online_backup = st.selectbox("Online Backup", list(THREE_WAY_SERVICE.keys()))
    device_protection = st.selectbox("Device Protection", list(THREE_WAY_SERVICE.keys()))
    tech_support = st.selectbox("Tech Support", list(THREE_WAY_SERVICE.keys()))
    streaming_tv = st.selectbox("Streaming TV", list(THREE_WAY_SERVICE.keys()))
    streaming_movies = st.selectbox("Streaming Movies", list(THREE_WAY_SERVICE.keys()))
    contract = st.selectbox("Contract", list(CONTRACT_MAP.keys()))
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", list(PAYMENT_METHOD_MAP.keys()))

st.subheader("Billing")
bcol1, bcol2 = st.columns(2)
with bcol1:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
with bcol2:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=840.0, step=1.0)

# ---------------------------------------------------------------------------
# Build feature row in the exact order the models were trained on
# ---------------------------------------------------------------------------
raw_values = {
    "gender": GENDER_MAP[gender],
    "SeniorCitizen": BINARY_YES_NO[senior_citizen],
    "Partner": BINARY_YES_NO[partner],
    "Dependents": BINARY_YES_NO[dependents],
    "tenure": tenure,
    "PhoneService": BINARY_YES_NO[phone_service],
    "MultipleLines": MULTIPLE_LINES_MAP[multiple_lines],
    "InternetService": INTERNET_SERVICE_MAP[internet_service],
    "OnlineSecurity": THREE_WAY_SERVICE[online_security],
    "OnlineBackup": THREE_WAY_SERVICE[online_backup],
    "DeviceProtection": THREE_WAY_SERVICE[device_protection],
    "TechSupport": THREE_WAY_SERVICE[tech_support],
    "StreamingTV": THREE_WAY_SERVICE[streaming_tv],
    "StreamingMovies": THREE_WAY_SERVICE[streaming_movies],
    "Contract": CONTRACT_MAP[contract],
    "PaperlessBilling": BINARY_YES_NO[paperless_billing],
    "PaymentMethod": PAYMENT_METHOD_MAP[payment_method],
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

with st.expander("Show encoded feature vector"):
    st.dataframe(pd.DataFrame([raw_values])[FEATURES])

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Predict churn", type="primary"):
    try:
        X = np.array([[raw_values[f] for f in FEATURES]])
        X_scaled = scaler.transform(X)

        model = rf_model if model_choice == "Random Forest" else lr_model
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        churn_prob = proba[1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ Prediction: **Likely to churn** (probability: {churn_prob:.1%})")
        else:
            st.success(f"✅ Prediction: **Likely to stay** (probability of churn: {churn_prob:.1%})")

        st.progress(float(churn_prob))
        st.caption(f"Model used: {model_choice}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
