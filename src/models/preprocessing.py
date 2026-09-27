from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Columns that are identifiers, outcomes, or leakage-sensitive fields.
EXCLUDED_MODEL_COLUMNS = {
    "project_id",
    "performance_id",
    "delay_days",
    "delay_flag",
    "project_delay_target",
    "actual_finish_date",
    "actual_duration_days",
    "final_delay_days",
    "delay_next_30_days",
}

# Raw date columns are not directly suitable for Logistic Regression.
RAW_DATE_COLUMNS = {
    "date",
    "planned_start_date",
    "planned_end_date",
}

TARGET_COLUMN = "delay_next_30_days"


def get_model_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical predictor columns.

    Raw date columns and leakage-sensitive columns are excluded.
    """

    excluded = EXCLUDED_MODEL_COLUMNS | RAW_DATE_COLUMNS

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]

    return numeric_columns, categorical_columns


def build_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Numeric:
        median imputation + missing indicators + standardization

    Categorical:
        most-frequent imputation + one-hot encoding
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
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
                    sparse_output=True,
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