import pandas as pd

from src.data.generators.project_daily_performance_generator import (
    generate_project_daily_performance,
)


def test_project_daily_performance_generation():

    projects = pd.DataFrame(
        {
            "project_id": [
                "PRJ-001",
                "PRJ-002",
            ],
            "planned_start_date": [
                "2025-01-01",
                "2025-01-01",
            ],
            "planned_end_date": [
                "2025-01-10",
                "2025-01-15",
            ],
            "actual_start_date": [
                "2025-01-01",
                "2025-01-02",
            ],
            "actual_end_date": [
                "2025-01-12",
                "2025-01-18",
            ],
            "planned_duration_days": [
                9,
                14,
            ],
            "actual_duration_days": [
                11,
                16,
            ],
            "contract_value": [
                1_000_000,
                2_000_000,
            ],
            "project_complexity": [
                "MEDIUM",
                "HIGH",
            ],
            "delay_days": [
                2,
                4,
            ],
            "delay_flag": [
                0,
                1,
            ],
        }
    )

    activities = pd.DataFrame(
        {
            "activity_id": [
                "ACT-001",
                "ACT-002",
                "ACT-003",
            ],
            "project_id": [
                "PRJ-001",
                "PRJ-001",
                "PRJ-002",
            ],
            "planned_start_date": [
                "2025-01-01",
                "2025-01-03",
                "2025-01-01",
            ],
            "planned_finish_date": [
                "2025-01-05",
                "2025-01-08",
                "2025-01-07",
            ],
            "progress_pct": [
                50,
                70,
                30,
            ],
            "critical_path_flag": [
                1,
                0,
                1,
            ],
            "delay_days": [
                0,
                2,
                3,
            ],
            "delay_flag": [
                0,
                1,
                1,
            ],
            "baseline_cost": [
                10_000,
                20_000,
                15_000,
            ],
            "actual_cost": [
                11_000,
                23_000,
                17_000,
            ],
        }
    )

    deliveries = pd.DataFrame(
        {
            "project_id": [
                "PRJ-001",
                "PRJ-002",
            ],
            "material_id": [
                "MAT-001",
                "MAT-002",
            ],
            "actual_delivery_date": [
                "2025-01-02",
                "2025-01-03",
            ],
            "delivered_quantity": [
                100,
                150,
            ],
        }
    )

    material_consumption = pd.DataFrame(
        {
            "date": [
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
            ],
            "project_id": [
                "PRJ-001",
                "PRJ-001",
                "PRJ-002",
            ],
            "material_id": [
                "MAT-001",
                "MAT-001",
                "MAT-002",
            ],
            "planned_daily_quantity": [
                10,
                10,
                15,
            ],
            "actual_daily_quantity": [
                8,
                12,
                14,
            ],
            "available_delivered_quantity": [
                100,
                100,
                150,
            ],
            "stockout_flag": [
                0,
                0,
                0,
            ],
        }
    )

    equipment_events = pd.DataFrame(
        {
            "event_date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
            ],
            "equipment_id": [
                "EQP-001",
                "EQP-001",
                "EQP-002",
            ],
            "operating_hours": [
                8,
                7,
                9,
            ],
            "downtime_hours": [
                1,
                2,
                0,
            ],
            "maintenance_flag": [
                0,
                1,
                0,
            ],
            "failure_flag": [
                0,
                0,
                1,
            ],
            "maintenance_cost": [
                100,
                500,
                1000,
            ],
        }
    )

    result = generate_project_daily_performance(
        projects=projects,
        activities=activities,
        deliveries=deliveries,
        material_consumption=material_consumption,
        equipment_events=equipment_events,
        seed=42,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert not result.empty

    assert result[
        "performance_id"
    ].is_unique

    assert result[
        "project_id"
    ].isin(
        projects["project_id"]
    ).all()

    assert result[
        "date"
    ].notna().all()

    assert (
        result[
            "planned_progress_pct"
        ]
        .between(0, 100)
        .all()
    )

    assert result[
        "delay_signal_flag"
    ].isin([0, 1]).all()

    assert result[
        "project_delay_target"
    ].isin([0, 1]).all()

    assert (
        result[
            "operational_risk_score"
        ]
        .between(0, 1)
        .all()
    )