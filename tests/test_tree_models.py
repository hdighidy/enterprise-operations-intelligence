import pandas as pd
import pytest

from src.models.tree_models import (
    evaluate_model,
)


class DummyModel:

    def predict(self, X):
        return [0, 1, 1, 0]

    def predict_proba(self, X):
        return [
            [0.8, 0.2],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.7, 0.3],
        ]


def test_evaluate_model_returns_required_metrics():

    model = DummyModel()

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4]
        }
    )

    y = [0, 1, 1, 0]

    metrics = evaluate_model(
        model,
        X,
        y,
    )

    expected_metrics = {
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }

    assert set(metrics.keys()) == expected_metrics


def test_metrics_are_between_zero_and_one():

    model = DummyModel()

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4]
        }
    )

    y = [0, 1, 1, 0]

    metrics = evaluate_model(
        model,
        X,
        y,
    )

    for value in metrics.values():
        assert 0 <= value <= 1


def test_evaluation_uses_binary_target():

    model = DummyModel()

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4]
        }
    )

    y = [0, 1, 1, 0]

    metrics = evaluate_model(
        model,
        X,
        y,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_model_output_is_numeric():

    model = DummyModel()

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4]
        }
    )

    y = [0, 1, 1, 0]

    metrics = evaluate_model(
        model,
        X,
        y,
    )

    assert all(
        isinstance(value, float)
        for value in metrics.values()
    )


def test_target_is_binary():

    target = pd.Series(
        [0, 1, 1, 0, 1]
    )

    assert set(target.unique()).issubset(
        {0, 1}
    )