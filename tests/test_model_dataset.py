import pandas as pd

from src.models.dataset import (
    prepare_features_and_target,
)

from src.models.split import (
    time_based_split,
)


def make_data():

    dates = pd.date_range(
        "2025-01-01",
        periods=20,
        freq="D",
    )

    return pd.DataFrame(
        {
            "project_id": [
                "P1"
            ] * 20,

            "performance_id": [
                f"PERF-{i}"
                for i in range(20)
            ],

            "date": dates,

            "active_activity_count": range(20),

            "activity_delay_days": [
                0
            ] * 20,

            "delay_days": [
                0
            ] * 20,

            "delay_flag": [
                0
            ] * 20,

            "operational_risk_score": [
                0.1
            ] * 20,

            "delay_signal_flag": [
                0
            ] * 20,

            "delay_next_30_days": [
                0,
                1,
            ] * 10,

            "project_delay_target": [
                0
            ] * 20,

            "actual_finish_date": dates,

            "actual_duration_days": [
                20
            ] * 20,

            "final_delay_days": [
                0
            ] * 20,
        }
    )


def test_target_is_separated():

    df = make_data()

    X, y = prepare_features_and_target(df)

    assert "delay_next_30_days" not in X.columns

    assert len(y) == len(df)


def test_identifiers_are_excluded():

    df = make_data()

    X, _ = prepare_features_and_target(df)

    assert "project_id" not in X.columns

    assert "performance_id" not in X.columns


def test_leakage_columns_are_excluded():

    df = make_data()

    X, _ = prepare_features_and_target(df)

    forbidden = {
        "delay_days",
        "delay_flag",
        "project_delay_target",
        "actual_finish_date",
        "actual_duration_days",
        "final_delay_days",
    }

    assert forbidden.isdisjoint(
        X.columns
    )


def test_temporal_split():

    df = make_data()

    train, validation, test = (
        time_based_split(df)
    )

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0

    assert (
        train["date"].max()
        <
        validation["date"].min()
    )

    assert (
        validation["date"].max()
        <
        test["date"].min()
    )


def test_split_preserves_all_records():

    df = make_data()

    train, validation, test = (
        time_based_split(df)
    )

    assert (
        len(train)
        + len(validation)
        + len(test)
        == len(df)
    )