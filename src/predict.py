from __future__ import annotations

import pickle
from pathlib import Path

from src.preprocessing import prepare_customer_features


MODEL_PATH = Path("models/best_model.pkl")


def load_model_artifact(path: str | Path = MODEL_PATH) -> dict:
    with open(path, "rb") as file:
        return pickle.load(file)


def predict_churn(customer: dict, artifact: dict) -> dict:
    features = prepare_customer_features(customer, artifact["feature_columns"])
    probability = float(artifact["model"].predict_proba(features)[:, 1][0])
    prediction = int(probability >= artifact.get("threshold", 0.5))

    if probability < 0.35:
        risk_category = "Low"
    elif probability < 0.65:
        risk_category = "Medium"
    else:
        risk_category = "High"

    return {
        "prediction": prediction,
        "churn_probability": probability,
        "risk_category": risk_category,
    }
