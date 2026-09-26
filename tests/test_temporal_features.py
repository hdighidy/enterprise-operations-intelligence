import pandas as pd
import pytest

from src.features.temporal_features import (
    build_temporal_features,
)


def make_test_data():
    records = []

    for project_id in ["PRJ-001", "PRJ-002"]:

        for day in range(10):

            records.append(
                {
                    "project_id": project_id,
                    "date": pd.Timestamp("2025-01-01")
                    + pd.Timedelta(days=day),

                    "active_activity_count": day + 1,
                    "critical_activity_count": 1,
                    "delayed_activity_count": day,
                    "activity_delay_days": day * 2,

                    "material_stockout_count": day % 3,
                    "delivery_count": day + 2,

                    "equipment_downtime_hours": float(day),
                    "equipment_failure_count": day % 2,
                    "equipment_maintenance_cost": float(day * 100),

                    "cost_variance": float(day * 1000),
                    "cost_variance_pct": float(day),
                }
            )

    return pd.DataFrame(records)


def test_temporal_data_is_sorted():
    df = make_test_data()

    result = build_temporal_features(df)

    for _, group in result.groupby("project_id"):
        dates = group["date"].tolist()

        assert dates == sorted(dates)


def test_lag_1_uses_previous_day():
    df = make_test_data()

    result = build_temporal_features(df)

    project = result[
        result["project_id"] == "PRJ-001"
    ].reset_index(drop=True)

    assert pd.isna(
        project.loc[0, "active_activity_count_lag_1"]
    )

    assert (
        project.loc[1, "active_activity_count_lag_1"]
        == project.loc[0, "active_activity_count"]
    )


def test_lag_7_uses_previous_seven_days():
    df = make_test_data()

    result = build_temporal_features(df)

    project = result[
        result["project_id"] == "PRJ-001"
    ].reset_index(drop=True)

    assert pd.isna(
        project.loc[6, "active_activity_count_lag_7"]
    )

    assert (
        project.loc[7, "active_activity_count_lag_7"]
        == project.loc[0, "active_activity_count"]
    )


def test_rolling_feature_does_not_use_current_day():
    df = make_test_data()

    result = build_temporal_features(df)

    project = result[
        result["project_id"] == "PRJ-001"
    ].reset_index(drop=True)

    # At day 1, the historical 7-day window contains
    # only day 0.
    assert (
        project.loc[1, "activity_delay_days_rolling_7d"]
        == project.loc[0, "activity_delay_days"]
    )


def test_projects_do_not_mix_features():
    df = make_test_data()

    result = build_temporal_features(df)

    project_1 = result[
        result["project_id"] == "PRJ-001"
    ].reset_index(drop=True)

    project_2 = result[
        result["project_id"] == "PRJ-002"
    ].reset_index(drop=True)

    assert (
        project_1.loc[1, "active_activity_count_lag_1"]
        == project_1.loc[0, "active_activity_count"]
    )

    assert (
        project_2.loc[1, "active_activity_count_lag_1"]
        == project_2.loc[0, "active_activity_count"]
    )


def test_no_future_value_is_used():
    df = make_test_data()

    result = build_temporal_features(df)

    project = result[
        result["project_id"] == "PRJ-001"
    ].reset_index(drop=True)

    # Day 5's lag must come from day 4,
    # never from day 6.
    assert (
        project.loc[5, "active_activity_count_lag_1"]
        == project.loc[4, "active_activity_count"]
    )

    assert (
        project.loc[5, "active_activity_count_lag_1"]
        != project.loc[6, "active_activity_count"]
    )