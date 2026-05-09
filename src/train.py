from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from src.evaluate import evaluate_classifier, metrics_table
from src.preprocessing import load_raw_data, make_model_frame, split_features_target


def train_best_model(output_path: str | Path = "models/best_model.pkl") -> dict:
    """Train a compact production-style artifact used by the Streamlit app."""
    model_frame = make_model_frame(load_raw_data())
    x, y = split_features_target(model_frame)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    sampler = SMOTEENN(random_state=42)
    x_train_resampled, y_train_resampled = sampler.fit_resample(x_train, y_train)

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [150, 250, 350],
            "max_depth": [5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced"],
        },
        n_iter=12,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(x_train_resampled, y_train_resampled)

    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        },
        n_iter=12,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    xgb_search.fit(x_train_resampled, y_train_resampled)

    candidates = {
        "Random Forest Tuned": rf_search.best_estimator_,
        "XGBoost Tuned": xgb_search.best_estimator_,
    }
    results = [
        evaluate_classifier(model, x_test, y_test, name)
        for name, model in candidates.items()
    ]
    comparison = metrics_table(results)
    best_name = comparison.iloc[0]["model"]
    best_model = candidates[best_name]

    feature_importance = pd.DataFrame(
        {
            "feature": x.columns,
            "importance": getattr(best_model, "feature_importances_", [0] * x.shape[1]),
        }
    ).sort_values("importance", ascending=False)

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": list(x.columns),
        "metrics": comparison.to_dict(orient="records"),
        "feature_importance": feature_importance.head(20).to_dict(orient="records"),
        "threshold": 0.5,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(artifact, file)

    return artifact


if __name__ == "__main__":
    train_best_model()
