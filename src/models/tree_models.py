from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

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


INPUT_PATH = Path(
    "data/processed/ml_dataset.csv"
)

RANDOM_STATE = 42


def build_decision_tree_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """
    Build a Decision Tree classification pipeline.
    """

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        model_family="tree",
    )

    classifier = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=6,
        min_samples_leaf=20,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_random_forest_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """
    Build a Random Forest classification pipeline.
    """

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        model_family="tree",
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def print_metrics(
    model_name: str,
    metrics: dict,
) -> None:
    """
    Print model evaluation metrics.
    """

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

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

    print()
    print("Confusion Matrix")
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


def run_tree_models(
    input_path: Path = INPUT_PATH,
) -> dict:
    """
    Train and evaluate tree-based baseline models.

    The test set remains locked and is not evaluated here.
    """

    df = load_ml_dataset(
        input_path
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = time_based_split(
        df,
        date_column="date",
    )

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if validation_df.empty:
        raise ValueError(
            "Validation dataset is empty."
        )

    if test_df.empty:
        raise ValueError(
            "Test dataset is empty."
        )

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

    numeric_columns, categorical_columns = (
        get_model_columns(X_train)
    )

    print("=" * 70)
    print("STEP 10.5B — TREE MODELING")
    print("=" * 70)

    print(
        f"Training records   : {len(X_train):,}"
    )

    print(
        f"Validation records : {len(X_validation):,}"
    )

    print(
        f"Test records       : {len(test_df):,}"
    )

    print(
        f"Numeric features   : {len(numeric_columns):,}"
    )

    print(
        f"Categorical        : {len(categorical_columns):,}"
    )

    print()
    print("Target rates:")

    print(
        f"Train       : {y_train.mean() * 100:.2f}%"
    )

    print(
        f"Validation  : {y_validation.mean() * 100:.2f}%"
    )

    print(
        "\nTest set remains locked."
    )

    # ---------------------------------------------------------
    # Decision Tree
    # ---------------------------------------------------------

    decision_tree = (
        build_decision_tree_pipeline(
            numeric_columns,
            categorical_columns,
        )
    )

    decision_tree.fit(
        X_train,
        y_train,
    )

    decision_tree_metrics = (
        evaluate_classifier(
            decision_tree,
            X_validation,
            y_validation,
        )
    )

    print_metrics(
        "Decision Tree — Validation",
        decision_tree_metrics,
    )

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    random_forest = (
        build_random_forest_pipeline(
            numeric_columns,
            categorical_columns,
        )
    )

    random_forest.fit(
        X_train,
        y_train,
    )

    random_forest_metrics = (
        evaluate_classifier(
            random_forest,
            X_validation,
            y_validation,
        )
    )

    print_metrics(
        "Random Forest — Validation",
        random_forest_metrics,
    )

    return {
        "decision_tree": decision_tree_metrics,
        "random_forest": random_forest_metrics,
    }


if __name__ == "__main__":
    run_tree_models()