from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(model, x_test, y_test, model_name: str) -> dict:
    """Evaluate a classifier with metrics that matter for churn modeling."""
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = model.decision_function(x_test)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def metrics_table(results: list[dict]) -> pd.DataFrame:
    """Build a clean comparison table from evaluation result dictionaries."""
    metric_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    return (
        pd.DataFrame(results)[metric_cols]
        .sort_values(["roc_auc", "recall", "f1"], ascending=False)
        .reset_index(drop=True)
    )
