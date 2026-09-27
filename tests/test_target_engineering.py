import pandas as pd
import pytest

from src.features.target_engineering import (
    create_future_delay_target,
)


def make_test_data():

    dates = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="D",
    )

    return pd.DataFrame(
        {
            "project_id": ["PRJ-001"] * 10,
            "date": dates,
            "activity_delay_days": [
                0,
                0,
                0,
                5,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }
    )


def test_future_delay_is_detected():

    df = make_test_data()

    result = create_future_delay_target(
        df,
        horizon_days=3,
        delay_threshold_days=3,
    )

    # Delay occurs on day 4.
    # Days 1-3 should therefore see
    # a future delay within 3 days.

    assert result.loc[
        0,
        "delay_next_30_days",
    ] == 1

    assert result.loc[
        1,
        "delay_next_30_days",
    ] == 1

    assert result.loc[
        2,
        "delay_next_30_days",
    ] == 1


def test_current_day_is_not_used():

    df = pd.DataFrame(
        {
            "project_id": ["PRJ-001"],
            "date": [
                pd.Timestamp("2025-01-01")
            ],
            "activity_delay_days": [10],
        }
    )

    result = create_future_delay_target(
        df,
        horizon_days=3,
        delay_threshold_days=3,
    )

    # The only delay is today.
    # It must not create a future-delay target.

    assert (
        result.loc[
            0,
            "delay_next_30_days",
        ]
        == 0
    )


def test_projects_are_isolated():

    df = pd.DataFrame(
        {
            "project_id": [
                "PRJ-001",
                "PRJ-001",
                "PRJ-002",
                "PRJ-002",
            ],
            "date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-01",
                "2025-01-02",
            ],
            "activity_delay_days": [
                0,
                0,
                10,
                0,
            ],
        }
    )

    result = create_future_delay_target(df)

    project_1 = result[
        result["project_id"] == "PRJ-001"
    ]

    assert (
        project_1["delay_next_30_days"].sum()
        == 0
    )


def test_target_is_binary():

    df = make_test_data()

    result = create_future_delay_target(df)

    assert set(
        result["delay_next_30_days"].unique()
    ).issubset({0, 1})


def test_missing_column_raises_error():

    df = pd.DataFrame(
        {
            "project_id": ["PRJ-001"],
            "date": ["2025-01-01"],
        }
    )

    with pytest.raises(ValueError):

        create_future_delay_target(df)