import pandas as pd

from src.models.baseline import build_logistic_pipeline
from src.models.explainability import (
    extract_logistic_coefficients,
)
from src.models.preprocessing import get_model_columns


def create_test_dataset():
    return pd.DataFrame(
        {
            "active_activity_count": [
                10, 20, 15, 30, 25, 40
            ],
            "activity_delay_days": [
                0, 2, 1, 8, 10, 12
            ],
            "material_stockout_count": [
                0, 0, 1, 2, 3, 4
            ],
            "project_complexity": [
                "LOW",
                "LOW",
                "MEDIUM",
                "HIGH",
                "HIGH",
                "VERY_HIGH",
            ],
            "delay_next_30_days": [
                0, 0, 0, 1, 1, 1
            ],
        }
    )


def test_coefficients_can_be_extracted():
    df = create_test_dataset()

    X = df.drop(
        columns=["delay_next_30_days"]
    )

    y = df["delay_next_30_days"]

    numeric_columns, categorical_columns = (
        get_model_columns(X)
    )

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    coefficients = extract_logistic_coefficients(
        model
    )

    assert not coefficients.empty


def test_coefficient_schema_is_correct():
    df = create_test_dataset()

    X = df.drop(
        columns=["delay_next_30_days"]
    )

    y = df["delay_next_30_days"]

    numeric_columns, categorical_columns = (
        get_model_columns(X)
    )

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    coefficients = extract_logistic_coefficients(
        model
    )

    required_columns = {
        "feature",
        "coefficient",
        "absolute_coefficient",
        "direction",
    }

    assert required_columns.issubset(
        coefficients.columns
    )


def test_coefficient_count_matches_feature_names():
    df = create_test_dataset()

    X = df.drop(
        columns=["delay_next_30_days"]
    )

    y = df["delay_next_30_days"]

    numeric_columns, categorical_columns = (
        get_model_columns(X)
    )

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    coefficients = extract_logistic_coefficients(
        model
    )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    assert len(coefficients) == len(
        feature_names
    )


def test_coefficients_are_sorted_by_absolute_value():
    df = create_test_dataset()

    X = df.drop(
        columns=["delay_next_30_days"]
    )

    y = df["delay_next_30_days"]

    numeric_columns, categorical_columns = (
        get_model_columns(X)
    )

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    coefficients = extract_logistic_coefficients(
        model
    )

    absolute_values = (
        coefficients["absolute_coefficient"]
        .tolist()
    )

    assert absolute_values == sorted(
        absolute_values,
        reverse=True,
    )


def test_direction_matches_coefficient_sign():
    df = create_test_dataset()

    X = df.drop(
        columns=["delay_next_30_days"]
    )

    y = df["delay_next_30_days"]

    numeric_columns, categorical_columns = (
        get_model_columns(X)
    )

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    coefficients = extract_logistic_coefficients(
        model
    )

    for _, row in coefficients.iterrows():

        if row["coefficient"] > 0:
            assert row["direction"] == "positive"

        elif row["coefficient"] < 0:
            assert row["direction"] == "negative"

        else:
            assert row["direction"] == "neutral"