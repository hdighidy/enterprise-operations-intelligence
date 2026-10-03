from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Columns that must never enter a model
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Raw date columns
#
# Dates are currently excluded from the model matrix.
# If calendar features are required later, they should be engineered
# explicitly rather than one-hot encoding raw dates.
# ---------------------------------------------------------------------

RAW_DATE_COLUMNS = {
    "date",
    "planned_start_date",
    "planned_end_date",
    "actual_start_date",
}


def get_model_columns(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Return numeric and categorical predictor columns.

    This function is the single source of truth for determining
    which columns are eligible for model preprocessing.
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
    model_family: str = "linear",
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline for a model family.

    model_family:
        linear -> imputation + scaling + one-hot encoding
        tree   -> imputation + one-hot encoding
    """

    if model_family not in {"linear", "tree"}:
        raise ValueError(
            "model_family must be either 'linear' or 'tree'."
        )

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        )
    ]

    if model_family == "linear":
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
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