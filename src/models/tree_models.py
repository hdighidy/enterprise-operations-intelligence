from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.split import time_based_split


DATA_PATH = Path("data/processed/ml_dataset.csv")

OUTPUT_PATH = Path(
    "data/processed/tree_model_validation_results.csv"
)

TARGET_COLUMN = "delay_next_30_days"


def evaluate_model(model, X, y):
    """Evaluate a binary classification model."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y, probabilities)
        ),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
    }


def build_preprocessor(X):
    """Build preprocessing pipeline for tree models."""

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
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


def prepare_data(train_df, validation_df):
    """Prepare train and validation matrices."""

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = train_df[TARGET_COLUMN].astype(int)

    X_validation = validation_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_validation = validation_df[
        TARGET_COLUMN
    ].astype(int)

    # Date is used for splitting, not as a
    # predictive model feature.
    for dataframe in (
        X_train,
        X_validation,
    ):
        for column in [
            "date",
            "performance_id",
        ]:
            if column in dataframe.columns:
                dataframe.drop(
                    columns=[column],
                    inplace=True,
                )

    preprocessor = build_preprocessor(
        X_train
    )

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_validation_transformed = (
        preprocessor.transform(
            X_validation
        )
    )

    return (
        X_train_transformed,
        y_train,
        X_validation_transformed,
        y_validation,
    )


def main():

    print("=" * 70)
    print("STEP 10.5 — TREE-BASED MODELING")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"ML dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    # ---------------------------------------------------------
    # Time-based split
    # ---------------------------------------------------------

    train_df, validation_df, test_df = (
        time_based_split(
            df,
            date_column="date",
            train_ratio=0.70,
            validation_ratio=0.15,
        )
    )

    print()
    print("TIME-BASED SPLIT")
    print("-" * 70)

    print(
        f"Train records      : {len(train_df):,}"
    )
    print(
        f"Validation records : {len(validation_df):,}"
    )
    print(
        f"Test records       : {len(test_df):,}"
    )

    print(
        f"Train target       : "
        f"{train_df[TARGET_COLUMN].mean():.2%}"
    )

    print(
        f"Validation target  : "
        f"{validation_df[TARGET_COLUMN].mean():.2%}"
    )

    print(
        f"Test target        : "
        f"{test_df[TARGET_COLUMN].mean():.2%}"
    )

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = prepare_data(
        train_df,
        validation_df,
    )

    print()
    print(
        f"Transformed features : "
        f"{X_train.shape[1]}"
    )

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            random_state=42,
        ),
    }

    results = []

    # ---------------------------------------------------------
    # Train / validation
    # ---------------------------------------------------------

    for model_name, model in models.items():

        print()
        print("=" * 70)
        print(model_name)
        print("=" * 70)

        model.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        for metric, value in metrics.items():
            print(
                f"{metric:<12}: "
                f"{value:.4f}"
            )

    # ---------------------------------------------------------
    # Save validation results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()