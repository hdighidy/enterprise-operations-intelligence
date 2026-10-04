import pandas as pd

from src.features.prediction_time_audit import (
    audit_feature_names,
    audit_target_construction,
    audit_missing_values,
)


def test_forbidden_feature_is_detected():

    df = pd.DataFrame(
        {
            "project_id": ["P1"],
            "actual_finish_date": ["2026-01-01"],
        }
    )

    results = audit_feature_names(df)

    forbidden = [
        result
        for result in results
        if result.feature == "actual_finish_date"
    ]

    assert forbidden
    assert forbidden[0].status == "FAIL"
    assert forbidden[0].severity == "HIGH"


def test_current_delay_requires_semantic_review():

    df = pd.DataFrame(
        {
            "activity_delay_days": [0, 5, 10],
        }
    )

    results = audit_feature_names(df)

    review = [
        result
        for result in results
        if result.feature == "activity_delay_days"
    ]

    assert review
    assert review[0].status == "REVIEW"


def test_lag_feature_passes_temporal_audit():

    df = pd.DataFrame(
        {
            "activity_delay_days_lag_1": [0, 2, 4],
        }
    )

    results = audit_feature_names(df)

    result = results[0]

    assert result.status == "PASS"


def test_target_is_binary():

    df = pd.DataFrame(
        {
            "delay_next_30_days": [0, 1, 0, 1],
        }
    )

    results = audit_target_construction(df)

    assert results[0].status == "PASS"


def test_invalid_target_is_detected():

    df = pd.DataFrame(
        {
            "delay_next_30_days": [0, 1, 2],
        }
    )

    results = audit_target_construction(df)

    assert results[0].status == "FAIL"


def test_missing_values_are_reported():

    df = pd.DataFrame(
        {
            "feature_a": [1, None, 3],
        }
    )

    results = audit_missing_values(df)

    assert len(results) == 1
    assert results[0].status == "REVIEW"