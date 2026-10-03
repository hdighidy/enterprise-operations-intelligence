import pandas as pd

from src.models.preprocessing import (
    RAW_DATE_COLUMNS,
    EXCLUDED_MODEL_COLUMNS,
    get_model_columns,
    build_preprocessor,
)


def test_date_and_excluded_columns_are_not_model_features():
    df = pd.DataFrame(
        {
            "project_id": ["P1", "P2"],
            "performance_id": ["A", "B"],
            "date": ["2026-01-01", "2026-01-02"],
            "planned_start_date": ["2025-01-01", "2025-02-01"],
            "planned_end_date": ["2026-01-01", "2026-02-01"],
            "active_activity_count": [10, 20],
            "project_complexity": ["LOW", "HIGH"],
            "delay_next_30_days": [0, 1],
        }
    )

    numeric, categorical = get_model_columns(df)

    features = set(numeric + categorical)

    assert "date" not in features
    assert "planned_start_date" not in features
    assert "planned_end_date" not in features

    for column in EXCLUDED_MODEL_COLUMNS:
        assert column not in features


def test_numeric_and_categorical_columns_are_identified():
    df = pd.DataFrame(
        {
            "numeric_feature": [1.0, 2.0],
            "integer_feature": [1, 2],
            "category_feature": ["LOW", "HIGH"],
        }
    )

    numeric, categorical = get_model_columns(df)

    assert "numeric_feature" in numeric
    assert "integer_feature" in numeric
    assert "category_feature" in categorical


def test_preprocessor_can_fit_and_transform():
    df = pd.DataFrame(
        {
            "numeric_feature": [1.0, None, 3.0],
            "category_feature": ["LOW", "HIGH", "LOW"],
        }
    )

    numeric, categorical = get_model_columns(df)

    preprocessor = build_preprocessor(
        numeric,
        categorical,
    )

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0


def test_missing_numeric_value_is_handled():
    df = pd.DataFrame(
        {
            "numeric_feature": [1.0, None, 3.0],
        }
    )

    numeric, categorical = get_model_columns(df)

    preprocessor = build_preprocessor(
        numeric,
        categorical,
    )

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= 2


def test_actual_start_date_is_not_model_feature():
    df = pd.DataFrame({
        "actual_start_date": ["2025-01-01", "2025-02-01"],
        "numeric_feature": [1.0, 2.0],
        "project_complexity": ["LOW", "HIGH"],
    })

    numeric, categorical = get_model_columns(df)

    features = set(numeric + categorical)

    assert "actual_start_date" not in features

def test_tree_preprocessor_does_not_scale_numeric_features():
    numeric_columns = ["feature_a"]
    categorical_columns = ["category"]

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
        model_family="tree",
    )

    numeric_pipeline = next(
        transformer
        for name, transformer, columns
        in preprocessor.transformers
        if name == "numeric"
    )

    assert "scaler" not in numeric_pipeline.named_steps


def test_linear_preprocessor_scales_numeric_features():
    numeric_columns = ["feature_a"]
    categorical_columns = ["category"]

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
        model_family="linear",
    )

    numeric_pipeline = next(
        transformer
        for name, transformer, columns
        in preprocessor.transformers
        if name == "numeric"
    )

    assert "scaler" in numeric_pipeline.named_steps


def test_raw_dates_are_excluded():
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01"]
            ),
            "planned_start_date": pd.to_datetime(
                ["2026-01-01"]
            ),
            "planned_end_date": pd.to_datetime(
                ["2026-02-01"]
            ),
            "actual_start_date": pd.to_datetime(
                ["2026-01-02"]
            ),
            "numeric_feature": [1.0],
            "category": ["A"],
        }
    )

    numeric, categorical = get_model_columns(df)

    assert "date" not in numeric
    assert "date" not in categorical
    assert "planned_start_date" not in numeric
    assert "actual_start_date" not in numeric