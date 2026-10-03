import pandas as pd
import pytest

from sklearn.dummy import DummyClassifier

from src.models.evaluation import evaluate_classifier


def test_evaluation_returns_required_metrics():
    X = pd.DataFrame(
        {
            "feature": [0, 1, 0, 1],
        }
    )

    y = pd.Series([0, 1, 0, 1])

    model = DummyClassifier(
        strategy="prior"
    )

    model.fit(X, y)

    metrics = evaluate_classifier(
        model,
        X,
        y,
    )

    required = {
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

    assert required.issubset(metrics.keys())


def test_invalid_threshold_is_rejected():
    X = pd.DataFrame(
        {
            "feature": [0, 1, 0, 1],
        }
    )

    y = pd.Series([0, 1, 0, 1])

    model = DummyClassifier(
        strategy="prior"
    )

    model.fit(X, y)

    with pytest.raises(ValueError):
        evaluate_classifier(
            model,
            X,
            y,
            threshold=1.5,
        )


def test_metrics_are_valid():
    X = pd.DataFrame(
        {
            "feature": [0, 1, 0, 1],
        }
    )

    y = pd.Series([0, 1, 0, 1])

    model = DummyClassifier(
        strategy="prior"
    )

    model.fit(X, y)

    metrics = evaluate_classifier(
        model,
        X,
        y,
    )

    for metric in [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        assert 0.0 <= metrics[metric] <= 1.0