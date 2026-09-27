import numpy as np
import pandas as pd

from src.models.baseline import (
    build_logistic_pipeline,
    evaluate_model,
)
from src.models.preprocessing import get_model_columns


def create_test_dataset():
    """Create a small deterministic classification dataset."""

    return pd.DataFrame(
        {
            "active_activity_count": [10, 20, 15, 30, 25, 40],
            "activity_delay_days": [0, 2, 1, 8, 10, 12],
            "material_stockout_count": [0, 0, 1, 2, 3, 4],
            "equipment_downtime_hours": [2, 3, 5, 10, 12, 15],
            "project_complexity": [
                "LOW",
                "LOW",
                "MEDIUM",
                "HIGH",
                "HIGH",
                "VERY_HIGH",
            ],
            "delay_next_30_days": [0, 0, 0, 1, 1, 1],
        }
    )


def test_logistic_pipeline_can_be_created():
    df = create_test_dataset()

    X = df.drop(columns=["delay_next_30_days"])

    numeric_columns, categorical_columns = get_model_columns(X)

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    assert model is not None
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_logistic_pipeline_can_fit_and_predict():
    df = create_test_dataset()

    X = df.drop(columns=["delay_next_30_days"])
    y = df["delay_next_30_days"]

    numeric_columns, categorical_columns = get_model_columns(X)

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(predictions) == len(y)
    assert probabilities.shape == (len(y), 2)


def test_predicted_probabilities_are_valid():
    df = create_test_dataset()

    X = df.drop(columns=["delay_next_30_days"])
    y = df["delay_next_30_days"]

    numeric_columns, categorical_columns = get_model_columns(X)

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_evaluate_model_returns_required_metrics():
    df = create_test_dataset()

    X = df.drop(columns=["delay_next_30_days"])
    y = df["delay_next_30_days"]

    numeric_columns, categorical_columns = get_model_columns(X)

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    metrics = evaluate_model(
        model,
        X,
        y,
        "Test Model",
    )

    required_metrics = {
        "model",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }

    assert required_metrics.issubset(metrics.keys())


def test_evaluation_metrics_are_valid():
    df = create_test_dataset()

    X = df.drop(columns=["delay_next_30_days"])
    y = df["delay_next_30_days"]

    numeric_columns, categorical_columns = get_model_columns(X)

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    metrics = evaluate_model(
        model,
        X,
        y,
        "Test Model",
    )

    for metric_name in [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        assert 0.0 <= metrics[metric_name] <= 1.0