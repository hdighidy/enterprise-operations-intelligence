from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path(
    "data/processed/ml_dataset.csv"
)


TARGET_COLUMN = "delay_next_30_days"


EXCLUDED_COLUMNS = {
    # Identifiers
    "project_id",
    "performance_id",

    # Current/final delay outcomes
    "delay_days",
    "delay_flag",
    "project_delay_target",

    # Future information
    "actual_finish_date",
    "actual_duration_days",
    "final_delay_days",

    # Target
    TARGET_COLUMN,
}


def load_ml_dataset(
    input_path: Path = DEFAULT_INPUT_PATH,
) -> pd.DataFrame:
    """Load the validated ML dataset."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"ML dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(
            "ML dataset is empty."
        )

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
):
    """
    Separate predictors X from target y.

    The original dataframe is not modified.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    y = df[TARGET_COLUMN].copy()

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X = df[feature_columns].copy()

    return X, y


def prepare_model_dataset(
    input_path: Path = DEFAULT_INPUT_PATH,
):
    """
    Load and prepare the modeling dataset.
    """

    df = load_ml_dataset(input_path)

    X, y = prepare_features_and_target(df)

    return X, y