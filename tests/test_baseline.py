from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.baseline import (
    build_dummy_baseline,
    build_logistic_pipeline,
)
from src.models.evaluation import evaluate_classifier
from src.models.preprocessing import get_model_columns


TEST_DATA_PATH = Path("data/processed/ml_dataset.csv")


@pytest.fixture(scope="module")
def sample_data():
    df = pd.read_csv(TEST_DATA_PATH)

    # Keep the test lightweight while preserving enough
    # rows for the preprocessing and modeling pipeline.
    return df.sample(
        n=min(500, len(df)),
        random_state=42,
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def model_data(sample_data):
    target_column = "delay_next_30_days"

    X = sample_data.drop(columns=[target_column])
    y = sample_data[target_column]

    numeric_columns, categorical_columns = get_model_columns(X)

    return (
        X,
        y,
        numeric_columns,
        categorical_columns,
    )


def test_build_logistic_pipeline(model_data):
    _, _, numeric_columns, categorical_columns = model_data

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_build_dummy_baseline():
    model = build_dummy_baseline()

    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_logistic_pipeline_fit_and_predict(model_data):
    X, y, numeric_columns, categorical_columns = model_data

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_logistic_pipeline_probability_output(model_data):
    X, y, numeric_columns, categorical_columns = model_data

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_evaluate_classifier_returns_required_metrics(model_data):
    X, y, numeric_columns, categorical_columns = model_data

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    metrics = evaluate_classifier(
        model,
        X,
        y,
    )

    required_metrics = {
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "threshold",
    }

    assert required_metrics.issubset(metrics.keys())


def test_evaluate_classifier_metrics_are_valid(model_data):
    X, y, numeric_columns, categorical_columns = model_data

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    metrics = evaluate_classifier(
        model,
        X,
        y,
    )

    probability_metrics = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    for metric in probability_metrics:
        assert 0.0 <= metrics[metric] <= 1.0


def test_evaluate_classifier_rejects_invalid_threshold(model_data):
    X, y, numeric_columns, categorical_columns = model_data

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(X, y)

    with pytest.raises(ValueError):
        evaluate_classifier(
            model,
            X,
            y,
            threshold=0.0,
        )

    with pytest.raises(ValueError):
        evaluate_classifier(
            model,
            X,
            y,
            threshold=1.0,
        )