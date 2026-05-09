from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict import load_model_artifact, predict_churn
from src.preprocessing import load_reference_customers


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; max-width: 1180px;}
    .metric-card {
        border: 1px solid #e7e9ee;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
    }
    .risk-low {color: #16803c; font-weight: 700;}
    .risk-medium {color: #b7791f; font-weight: 700;}
    .risk-high {color: #c53030; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifact():
    return load_model_artifact(MODEL_PATH)


@st.cache_data
def get_reference_data():
    return load_reference_customers(PROJECT_ROOT / "data" / "raw" / "first_telc.csv")


def risk_class(risk_category: str) -> str:
    return {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high",
    }.get(risk_category, "risk-medium")


st.title("Customer Churn Prediction & Retention Analytics")
st.caption("Predict churn probability, classify retention risk, and inspect the main churn drivers.")

try:
    artifact = get_artifact()
except FileNotFoundError:
    st.error(
        "Model artifact not found. Run `notebooks/02_model_training.ipynb` or `python -m src.train` "
        "to create `models/best_model.pkl`."
    )
    st.stop()

reference_df = get_reference_data()

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Prediction Form")
    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            gender = st.selectbox("Gender", sorted(reference_df["gender"].dropna().unique()))
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", sorted(reference_df["Partner"].dropna().unique()))
            dependents = st.selectbox("Dependents", sorted(reference_df["Dependents"].dropna().unique()))
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)

        with col_b:
            phone_service = st.selectbox("Phone Service", sorted(reference_df["PhoneService"].dropna().unique()))
            multiple_lines = st.selectbox("Multiple Lines", sorted(reference_df["MultipleLines"].dropna().unique()))
            internet_service = st.selectbox("Internet Service", sorted(reference_df["InternetService"].dropna().unique()))
            online_security = st.selectbox("Online Security", sorted(reference_df["OnlineSecurity"].dropna().unique()))
            online_backup = st.selectbox("Online Backup", sorted(reference_df["OnlineBackup"].dropna().unique()))
            device_protection = st.selectbox("Device Protection", sorted(reference_df["DeviceProtection"].dropna().unique()))

        with col_c:
            tech_support = st.selectbox("Tech Support", sorted(reference_df["TechSupport"].dropna().unique()))
            streaming_tv = st.selectbox("Streaming TV", sorted(reference_df["StreamingTV"].dropna().unique()))
            streaming_movies = st.selectbox("Streaming Movies", sorted(reference_df["StreamingMovies"].dropna().unique()))
            contract = st.selectbox("Contract", sorted(reference_df["Contract"].dropna().unique()))
            paperless_billing = st.selectbox("Paperless Billing", sorted(reference_df["PaperlessBilling"].dropna().unique()))
            payment_method = st.selectbox("Payment Method", sorted(reference_df["PaymentMethod"].dropna().unique()))

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=float(monthly_charges * tenure),
            step=10.0,
        )

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

with right:
    st.subheader("Prediction Result")
    if submitted:
        result = predict_churn(customer, artifact)
        probability = result["churn_probability"]
        risk = result["risk_category"]

        st.metric("Churn Probability", f"{probability:.1%}")
        st.progress(probability)
        st.markdown(
            f"Risk Category: <span class='{risk_class(risk)}'>{risk}</span>",
            unsafe_allow_html=True,
        )

        if risk == "High":
            st.warning("Recommended action: prioritize for retention outreach with a personalized offer.")
        elif risk == "Medium":
            st.info("Recommended action: monitor closely and target with service education or contract incentives.")
        else:
            st.success("Recommended action: maintain engagement and avoid unnecessary discounting.")
    else:
        st.info("Submit the form to generate a churn risk score.")

    st.divider()
    st.subheader("Model Summary")
    st.write(f"Active model: **{artifact.get('model_name', 'Best model')}**")
    st.write(f"Decision threshold: **{artifact.get('threshold', 0.5):.2f}**")

tabs = st.tabs(["Model Performance", "Feature Importance"])

with tabs[0]:
    metrics = pd.DataFrame(artifact.get("metrics", []))
    if not metrics.empty:
        display_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
        st.dataframe(
            metrics[display_cols].style.format(
                {
                    "accuracy": "{:.3f}",
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1": "{:.3f}",
                    "roc_auc": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Model metrics will appear after retraining the notebook artifact.")

with tabs[1]:
    importance = pd.DataFrame(artifact.get("feature_importance", []))
    if not importance.empty:
        st.bar_chart(importance.set_index("feature")["importance"])
        st.caption("Higher values indicate stronger influence in the selected model.")
    else:
        st.info("Feature importance will appear after retraining with the modern notebook.")
