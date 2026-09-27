"""
Step 10.5 — Nonlinear Model Benchmark

Benchmarks tree-based models against the existing Logistic Regression
baseline using the same leakage-controlled, time-based train/validation split.

Models:
    1. Decision Tree
    2. Random Forest
    3. HistGradientBoosting

IMPORTANT:
    The test set is intentionally NOT evaluated here.
    Model selection must be performed using the validation set first.
"""

from __future__ import annotations

from pathlib import Path
import time

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.preprocessing import get_model_columns


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_PATH = Path("data/processed/ml_dataset.csv")
SPLIT_PATH = Path("data/processed/time_based_split.csv")
OUTPUT_PATH = Path("data/processed/tree_model_validation_results.csv")

TARGET = "delay_next_30_days"


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

MODELS = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    ),
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ML dataset and time-based split definition."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"ML dataset not found: {DATA_PATH}"
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Time-based split not found: {SPLIT_PATH}"
        )

    data = pd.read_csv(DATA_PATH)
    split = pd.read_csv(SPLIT_PATH)

    data["date"] = pd.to_datetime(data["date"])
    split["date"] = pd.to_datetime(split["date"])

    return data, split


# ---------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------


def build_splits(
    data: pd.DataFrame,
    split: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct training and validation datasets using the existing
    time-based split.

    The split file must contain:
        date
        split
    """

    required_columns = {"date", "split"}

    missing = required_columns - set(split.columns)

    if missing:
        raise ValueError(
            f"Split file missing required columns: {sorted(missing)}"
        )

    split_map = split[["date", "split"]].drop_duplicates()

    data = data.merge(
        split_map,
        on="date",
        how="left",
        validate="many_to_one",
    )

    if data["split"].isna().any():
        raise ValueError(
            "Some ML dataset dates do not exist in the time-based split."
        )

    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()

    if train.empty:
        raise ValueError("Training split is empty.")

    if validation.empty:
        raise ValueError("Validation split is empty.")

    return train, validation


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------


def build_preprocessor(
    train_df: pd.DataFrame,
) -> ColumnTransformer:
    """Build preprocessing using the project's existing column logic."""

    numeric_columns, categorical_columns = get_model_columns(train_df)

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate_model(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict:
    """Evaluate a classifier on the validation set."""

    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
    ).ravel()

    return {
        "roc_auc": roc_auc_score(
            y_validation,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_validation,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_validation,
            predictions,
        ),
        "precision": precision_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 70)
    print("STEP 10.5 — NONLINEAR MODEL BENCHMARK")
    print("=" * 70)

    data, split = load_data()

    train_df, validation_df = build_splits(
        data,
        split,
    )

    X_train = train_df.drop(
        columns=[TARGET, "split"],
        errors="ignore",
    )

    y_train = train_df[TARGET]

    X_validation = validation_df.drop(
        columns=[TARGET, "split"],
        errors="ignore",
    )

    y_validation = validation_df[TARGET]

    print(f"Training records   : {len(train_df):,}")
    print(f"Validation records : {len(validation_df):,}")

    print(
        f"Training target    : {y_train.mean():.2%}"
    )

    print(
        f"Validation target  : {y_validation.mean():.2%}"
    )

    print("-" * 70)

    preprocessor = build_preprocessor(
        train_df
    )

    results = []

    for model_name, classifier in MODELS.items():

        print(f"\n{'=' * 70}")
        print(model_name)
        print("=" * 70)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    classifier,
                ),
            ]
        )

        start_time = time.perf_counter()

        pipeline.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            time.perf_counter() - start_time
        )

        metrics = evaluate_model(
            pipeline,
            X_validation,
            y_validation,
        )

        metrics["model"] = model_name
        metrics["training_seconds"] = training_seconds

        results.append(metrics)

        print(
            f"ROC-AUC     : {metrics['roc_auc']:.4f}"
        )
        print(
            f"PR-AUC      : {metrics['pr_auc']:.4f}"
        )
        print(
            f"Accuracy    : {metrics['accuracy']:.4f}"
        )
        print(
            f"Precision   : {metrics['precision']:.4f}"
        )
        print(
            f"Recall      : {metrics['recall']:.4f}"
        )
        print(
            f"F1          : {metrics['f1']:.4f}"
        )
        print(
            f"Training    : {training_seconds:.2f} sec"
        )

        print("\nConfusion Matrix:")
        print(
            f"[[{metrics['true_negative']:5d} "
            f"{metrics['false_positive']:5d}]"
        )
        print(
            f" [{metrics['false_negative']:5d} "
            f"{metrics['true_positive']:5d}]]"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "model",
            "roc_auc",
            "pr_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "training_seconds",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "roc_auc",
                "pr_auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )

    print("\nOutput:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()