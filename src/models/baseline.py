"""
Step 10.3 — Baseline Model

Establish a reproducible Logistic Regression baseline for predicting
whether a project will experience a delay within the next 30 days.

Architecture:
    ML Dataset
        ↓
    Time-Based Split
        ↓
    Feature / Target Contract
        ↓
    Linear Preprocessing
        ↓
    Logistic Regression
        ↓
    Validation Evaluation

Important:
    - The test set is intentionally NOT evaluated here.
    - Model selection and threshold tuning must use validation data.
    - The test set is reserved for final evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.dataset import (
    load_ml_dataset,
    prepare_features_and_target,
)
from src.models.evaluation import evaluate_classifier
from src.models.preprocessing import (
    build_preprocessor,
    get_model_columns,
)
from src.models.split import time_based_split


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_PATH = Path(
    "data/processed/ml_dataset.csv"
)

TARGET_COLUMN = "delay_next_30_days"

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def build_logistic_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """
    Build the Logistic Regression modeling pipeline.

    Linear-model preprocessing includes:

        Numeric:
            - median imputation
            - missing-value indicators
            - standard scaling

        Categorical:
            - most-frequent imputation
            - one-hot encoding

    Parameters
    ----------
    numeric_columns:
        Numeric predictor columns.

    categorical_columns:
        Categorical predictor columns.

    Returns
    -------
    Pipeline
        Complete preprocessing + Logistic Regression pipeline.
    """

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        model_family="linear",
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------------------
# Dummy baseline
# ---------------------------------------------------------------------

def build_dummy_baseline() -> DummyClassifier:
    """
    Build a simple prior-probability baseline.

    The DummyClassifier predicts according to the class distribution
    observed in the training data.

    This provides a reference point for determining whether the
    Logistic Regression model learns useful predictive structure.
    """

    return DummyClassifier(
        strategy="prior"
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_dataset_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    """
    Print the chronological train/validation dataset summary.
    """

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Training records   : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation records : "
        f"{len(validation_df):,}"
    )

    print()

    print(
        f"Training date      : "
        f"{train_df['date'].min().date()} "
        f"→ "
        f"{train_df['date'].max().date()}"
    )

    print(
        f"Validation date    : "
        f"{validation_df['date'].min().date()} "
        f"→ "
        f"{validation_df['date'].max().date()}"
    )

    print()

    print(
        f"Training target    : "
        f"{train_df[TARGET_COLUMN].mean():.2%}"
    )

    print(
        f"Validation target  : "
        f"{validation_df[TARGET_COLUMN].mean():.2%}"
    )


def print_model_metrics(
    model_name: str,
    metrics: dict,
) -> None:
    """
    Print model evaluation metrics in a consistent format.
    """

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        f"ROC-AUC     : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC      : "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Accuracy    : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall      : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1          : "
        f"{metrics['f1']:.4f}"
    )

    print()
    print("Confusion Matrix")
    print("-" * 30)

    print(
        f"TN: {metrics['true_negative']:,}"
    )

    print(
        f"FP: {metrics['false_positive']:,}"
    )

    print(
        f"FN: {metrics['false_negative']:,}"
    )

    print(
        f"TP: {metrics['true_positive']:,}"
    )


# ---------------------------------------------------------------------
# Main baseline experiment
# ---------------------------------------------------------------------

def run_baseline(
    input_path: Path = INPUT_PATH,
) -> dict:
    """
    Run the baseline modeling experiment.

    Workflow
    --------
    1. Load canonical ML dataset.
    2. Create chronological train/validation/test split.
    3. Use only train and validation for this experiment.
    4. Build canonical feature/target matrices.
    5. Build Logistic Regression preprocessing.
    6. Train DummyClassifier.
    7. Train Logistic Regression.
    8. Evaluate both on validation data.
    9. Return validation metrics.

    The test set is intentionally not evaluated.
    """

    print()
    print("=" * 70)
    print("STEP 10.3 — BASELINE MODELING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load canonical ML dataset
    # ---------------------------------------------------------------

    df = load_ml_dataset(
        input_path=input_path
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column "
            f"'{TARGET_COLUMN}' is missing."
        )

    # ---------------------------------------------------------------
    # 2. Chronological split
    # ---------------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = time_based_split(
        df,
        date_column="date",
    )

    # ---------------------------------------------------------------
    # 3. Informational test-set check
    #
    # We verify that a test split exists, but deliberately do not
    # evaluate it.
    # ---------------------------------------------------------------

    if test_df.empty:
        raise ValueError(
            "Test split is empty."
        )

    # ---------------------------------------------------------------
    # 4. Print dataset summary
    # ---------------------------------------------------------------

    print_dataset_summary(
        train_df,
        validation_df,
    )

    print()
    print(
        f"Test records       : "
        f"{len(test_df):,}"
    )

    print(
        f"Test date          : "
        f"{test_df['date'].min().date()} "
        f"→ "
        f"{test_df['date'].max().date()}"
    )

    # ---------------------------------------------------------------
    # 5. Prepare features and target
    # ---------------------------------------------------------------

    X_train, y_train = (
        prepare_features_and_target(
            train_df
        )
    )

    X_validation, y_validation = (
        prepare_features_and_target(
            validation_df
        )
    )

    # ---------------------------------------------------------------
    # 6. Determine feature types
    #
    # This is the canonical feature-column contract.
    # ---------------------------------------------------------------

    (
        numeric_columns,
        categorical_columns,
    ) = get_model_columns(
        X_train
    )

    print()
    print("=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)

    print(
        f"Numeric features   : "
        f"{len(numeric_columns):,}"
    )

    print(
        f"Categorical        : "
        f"{len(categorical_columns):,}"
    )

    print(
        f"Total predictors   : "
        f"{len(numeric_columns) + len(categorical_columns):,}"
    )

    # ---------------------------------------------------------------
    # 7. Dummy baseline
    # ---------------------------------------------------------------

    dummy_model = build_dummy_baseline()

    dummy_model.fit(
        X_train,
        y_train,
    )

    dummy_metrics = evaluate_classifier(
        dummy_model,
        X_validation,
        y_validation,
        threshold=0.5,
    )

    print_model_metrics(
        "DummyClassifier — Validation",
        dummy_metrics,
    )

    # ---------------------------------------------------------------
    # 8. Logistic Regression baseline
    # ---------------------------------------------------------------

    logistic_model = build_logistic_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_metrics = evaluate_classifier(
        logistic_model,
        X_validation,
        y_validation,
        threshold=0.5,
    )

    print_model_metrics(
        "Logistic Regression — Validation",
        logistic_metrics,
    )

    # ---------------------------------------------------------------
    # 9. Return experiment results
    # ---------------------------------------------------------------

    results = {
        "dummy_validation": dummy_metrics,
        "logistic_validation": logistic_metrics,
        "numeric_features": len(
            numeric_columns
        ),
        "categorical_features": len(
            categorical_columns
        ),
        "training_records": len(
            train_df
        ),
        "validation_records": len(
            validation_df
        ),
        "test_records": len(
            test_df
        ),
    }

    return results


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_baseline()