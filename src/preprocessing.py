from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
REFERENCE_DATA_PATH = Path("data/raw/first_telc.csv")
TENURE_LABELS = [f"{i} - {i + 11}" for i in range(1, 72, 12)]


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the original Telco churn dataset."""
    return pd.read_csv(path)


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the project preprocessing used in the original notebook."""
    cleaned = df.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned = cleaned.dropna(how="any")

    cleaned["tenure_group"] = pd.cut(
        cleaned["tenure"],
        range(1, 80, 12),
        right=False,
        labels=TENURE_LABELS,
    )

    columns_to_drop = [col for col in ["customerID", "tenure"] if col in cleaned.columns]
    cleaned = cleaned.drop(columns=columns_to_drop)
    return cleaned


def make_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the dummy-encoded frame used for model training."""
    cleaned = clean_telco_data(df)
    if "Churn" in cleaned.columns:
        cleaned["Churn"] = np.where(cleaned["Churn"] == "Yes", 1, 0)
    return pd.get_dummies(cleaned)


def split_features_target(model_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a prepared model frame into features and target."""
    return model_frame.drop("Churn", axis=1), model_frame["Churn"]


def prepare_customer_features(
    customer: dict,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transform one app/customer record into the trained model feature layout."""
    customer_df = pd.DataFrame([customer])
    customer_df["TotalCharges"] = pd.to_numeric(customer_df["TotalCharges"], errors="coerce")
    customer_df["tenure"] = pd.to_numeric(customer_df["tenure"], errors="coerce")
    customer_df["MonthlyCharges"] = pd.to_numeric(customer_df["MonthlyCharges"], errors="coerce")
    customer_df["SeniorCitizen"] = pd.to_numeric(customer_df["SeniorCitizen"], errors="coerce").astype(int)

    customer_df["tenure_group"] = pd.cut(
        customer_df["tenure"],
        range(1, 80, 12),
        right=False,
        labels=TENURE_LABELS,
    )
    customer_df = customer_df.drop(columns=["tenure"])

    encoded = pd.get_dummies(customer_df)
    return encoded.reindex(columns=feature_columns, fill_value=0)


def load_reference_customers(path: str | Path = REFERENCE_DATA_PATH) -> pd.DataFrame:
    """Load a small raw sample used for app defaults and value discovery."""
    reference = pd.read_csv(path)
    return reference.drop(columns=[col for col in ["Unnamed: 0"] if col in reference.columns])
