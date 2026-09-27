from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.models.dataset import load_ml_dataset, prepare_features_and_target
from src.models.preprocessing import build_preprocessor, get_model_columns
from src.models.split import time_based_split


INPUT_PATH = Path("data/processed/ml_dataset.csv")


def evaluate_model(
    model,
    X,
    y,
    model_name: str,
) -> dict:
    """Evaluate a binary classification model."""

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": model_name,
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
    }

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    for metric, value in metrics.items():
        if metric == "model":
            continue

        print(f"{metric:12s}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))

    return metrics


def build_logistic_pipeline(
    numeric_columns,
    categorical_columns,
):
    """Build preprocessing + Logistic Regression pipeline."""

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def run_baseline():
    """Run baseline models using the temporal split."""

    df = load_ml_dataset(INPUT_PATH)

    train_df, validation_df, test_df = time_based_split(
        df,
        date_column="date",
    )

    X_train, y_train = prepare_features_and_target(train_df)
    X_validation, y_validation = prepare_features_and_target(
        validation_df
    )
    X_test, y_test = prepare_features_and_target(test_df)

    numeric_columns, categorical_columns = get_model_columns(
        X_train
    )

    print("=" * 70)
    print("STEP 10.3 — BASELINE MODELING")
    print("=" * 70)

    print(f"Training records   : {len(X_train):,}")
    print(f"Validation records : {len(X_validation):,}")
    print(f"Test records       : {len(X_test):,}")

    print(f"Numeric features   : {len(numeric_columns):,}")
    print(f"Categorical        : {len(categorical_columns):,}")

    print("\nTarget rates:")
    print(
        f"Train       : {y_train.mean() * 100:.2f}%"
    )
    print(
        f"Validation  : {y_validation.mean() * 100:.2f}%"
    )
    print(
        f"Test        : {y_test.mean() * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # Baseline 0 — Majority Class
    # ------------------------------------------------------------

    dummy = DummyClassifier(
        strategy="prior"
    )

    dummy.fit(X_train, y_train)

    dummy_validation = evaluate_model(
        dummy,
        X_validation,
        y_validation,
        "DummyClassifier",
    )

    # ------------------------------------------------------------
    # Baseline 1 — Logistic Regression
    # ------------------------------------------------------------

    logistic_model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_validation = evaluate_model(
        logistic_model,
        X_validation,
        y_validation,
        "Logistic Regression — Validation",
    )

    logistic_test = evaluate_model(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression — Test",
    )

    return {
        "dummy_validation": dummy_validation,
        "logistic_validation": logistic_validation,
        "logistic_test": logistic_test,
    }


if __name__ == "__main__":
    run_baseline()