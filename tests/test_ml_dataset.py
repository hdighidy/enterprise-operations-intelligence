import pandas as pd

from src.features.build_ml_dataset import (
    build_ml_dataset,
)


def make_test_data():

    dates = pd.date_range(
        "2025-01-01",
        periods=40,
        freq="D",
    )

    records = []

    for project_id in [
        "PRJ-001",
        "PRJ-002",
    ]:

        for i, date in enumerate(dates):

            records.append(
                {
                    "project_id": project_id,
                    "date": date,

                    "active_activity_count": 10,
                    "critical_activity_count": 2,
                    "delayed_activity_count": i % 3,
                    "activity_delay_days": (
                        5 if i == 20 else 0
                    ),

                    "material_stockout_count": i % 2,
                    "delivery_count": 5,

                    "equipment_downtime_hours": 2.0,
                    "equipment_failure_count": 0,
                    "equipment_maintenance_cost": 100.0,

                    "cost_variance": 1000.0,
                    "cost_variance_pct": 1.0,

                    # Deliberate leakage columns.
                    "project_delay_target": 1,
                    "actual_finish_date": date,
                    "actual_duration_days": 40,
                    "final_delay_days": 5,
                }
            )

    return pd.DataFrame(records)


def test_ml_dataset_creation(tmp_path):

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "ml_dataset.csv"

    df = make_test_data()

    df.to_csv(
        input_path,
        index=False,
    )

    result = build_ml_dataset(
        input_path=input_path,
        output_path=output_path,
    )

    assert len(result) == len(df)

    assert result["project_id"].nunique() == 2

    assert "delay_next_30_days" in result.columns

    assert set(
        result["delay_next_30_days"].unique()
    ).issubset({0, 1})

    assert output_path.exists()


def test_leakage_columns_removed(tmp_path):

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "ml_dataset.csv"

    df = make_test_data()

    df.to_csv(
        input_path,
        index=False,
    )

    result = build_ml_dataset(
        input_path=input_path,
        output_path=output_path,
    )

    forbidden = {
        "project_delay_target",
        "actual_finish_date",
        "actual_duration_days",
        "final_delay_days",
    }

    assert forbidden.isdisjoint(
        result.columns
    )


def test_temporal_order_is_preserved(tmp_path):

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "ml_dataset.csv"

    df = make_test_data()

    # Deliberately shuffle input.
    df = df.sample(
        frac=1,
        random_state=42,
    )

    df.to_csv(
        input_path,
        index=False,
    )

    result = build_ml_dataset(
        input_path=input_path,
        output_path=output_path,
    )

    for _, group in result.groupby(
        "project_id"
    ):

        dates = group["date"].tolist()

        assert dates == sorted(dates)