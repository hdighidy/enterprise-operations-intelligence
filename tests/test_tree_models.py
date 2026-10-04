import numpy as np
import pandas as pd

from src.models.evaluation import evaluate_classifier
from src.models.tree_models import (
    build_decision_tree_pipeline,
    build_random_forest_pipeline,
)


class DummyModel:

    def predict(self, X):
        return [0, 1, 1, 0]

    def predict_proba(self, X):
        return [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.8, 0.2],
        ]


def test_evaluate_model_returns_required_metrics():

    model = DummyModel()

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4]
        }
    )

    y = [0, 1, 1, 0]

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

    assert required_metrics.issubset(
        metrics.keys()
    )


def test_metrics_are_between_zero_and_one():

    model = DummyModel()

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4]
        }
    )

    y = [0, 1, 1, 0]

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


def test_evaluation_uses_binary_target():

    model = DummyModel()

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4]
        }
    )

    y = [0, 1, 1, 0]

    metrics = evaluate_classifier(
        model,
        X,
        y,
    )

    assert (
        metrics["true_negative"]
        + metrics["false_positive"]
        + metrics["false_negative"]
        + metrics["true_positive"]
        == len(y)
    )


def test_model_output_is_numeric():

    model = DummyModel()

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4]
        }
    )

    y = [0, 1, 1, 0]

    metrics = evaluate_classifier(
        model,
        X,
        y,
    )

    assert isinstance(
        metrics["roc_auc"],
        float,
    )

    assert isinstance(
        metrics["f1"],
        float,
    )


def test_decision_tree_pipeline_creation():

    model = build_decision_tree_pipeline(
        numeric_columns=["feature_a"],
        categorical_columns=["category"],
    )

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

    assert (
        model.named_steps["classifier"]
        .__class__.__name__
        == "DecisionTreeClassifier"
    )


def test_random_forest_pipeline_creation():

    model = build_random_forest_pipeline(
        numeric_columns=["feature_a"],
        categorical_columns=["category"],
    )

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

    assert (
        model.named_steps["classifier"]
        .__class__.__name__
        == "RandomForestClassifier"
    )