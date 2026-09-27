"""
Tests for Step 10.5 — Nonlinear Model Benchmark.
"""

import pandas as pd
import pytest

from src.models.tree_models import (
    build_preprocessor,
    build_splits,
    evaluate_model,
)


def test_build_splits_separates_train_and_validation():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                ]
            ),
            "feature": [1.0, 2.0, 3.0, 4.0],
            "delay_next_30_days": [0, 1, 0, 1],
        }
    )

    split = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                ]
            ),
            "split": [
                "train",
                "train",
                "validation",
                "validation",
            ],
        }
    )

    train, validation = build_splits(
        data,
        split,
    )

    assert len(train) == 2
    assert len(validation) == 2

    assert set(train["date"]) == {
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
    }

    assert set(validation["date"]) == {
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-04"),
    }


def test_build_splits_requires_split_columns():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01"]
            ),
            "delay_next_30_days": [1],
        }
    )

    invalid_split = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01"]
            )
        }
    )

    with pytest.raises(ValueError):
        build_splits(
            data,
            invalid_split,
        )


def test_preprocessor_can_be_created():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-02"]
            ),
            "actual_start_date": pd.to_datetime(
                ["2025-01-01", "2025-01-02"]
            ),
            "numeric_feature": [1.0, 2.0],
            "project_complexity": [
                "LOW",
                "HIGH",
            ],
            "delay_next_30_days": [0, 1],
        }
    )

    preprocessor = build_preprocessor(
        data
    )

    assert preprocessor is not None


def test_evaluate_model_returns_required_metrics():
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    data = pd.DataFrame(
        {
            "feature": [0, 1, 2, 3, 4, 5],
        }
    )

    target = pd.Series(
        [0, 1, 0, 1, 0, 1]
    )

    model = Pipeline(
        steps=[
            (
                "model",
                DummyClassifier(
                    strategy="prior"
                ),
            )
        ]
    )

    model.fit(
        data,
        target,
    )

    metrics = evaluate_model(
        model,
        data,
        target,
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
    }

    assert required.issubset(
        metrics.keys()
    )


def test_evaluation_metrics_are_bounded():
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    data = pd.DataFrame(
        {
            "feature": [0, 1, 2, 3, 4, 5],
        }
    )

    target = pd.Series(
        [0, 1, 0, 1, 0, 1]
    )

    model = Pipeline(
        steps=[
            (
                "model",
                DummyClassifier(
                    strategy="prior"
                ),
            )
        ]
    )

    model.fit(
        data,
        target,
    )

    metrics = evaluate_model(
        model,
        data,
        target,
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