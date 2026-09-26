import pandas as pd

from src.data.generators.equipment_event_generator import (
    generate_equipment_events,
)


def test_equipment_event_generation():

    equipment = pd.DataFrame(
        {
            "equipment_id": [
                "EQP-001",
                "EQP-002",
                "EQP-003",
            ],
            "equipment_type": [
                "Crane",
                "Excavator",
                "Welding Machine",
            ],
            "operating_hours": [
                5000,
                8000,
                3000,
            ],
            "downtime_hours": [
                20.0,
                35.0,
                15.0,
            ],
            "utilization_rate": [
                0.80,
                0.75,
                0.65,
            ],
            "maintenance_cost": [
                5000.0,
                7000.0,
                3000.0,
            ],
        }
    )

    events = generate_equipment_events(
        equipment=equipment,
        seed=42,
        max_equipment=3,
        days=30,
    )

    assert isinstance(
        events,
        pd.DataFrame,
    )

    assert not events.empty

    assert events[
        "event_id"
    ].is_unique

    assert events[
        "equipment_id"
    ].isin(
        equipment["equipment_id"]
    ).all()

    assert (
        events["operating_hours"]
        >= 0
    ).all()

    assert (
        events["downtime_hours"]
        >= 0
    ).all()

    assert (
        events["maintenance_cost"]
        >= 0
    ).all()

    assert events[
        "maintenance_flag"
    ].isin([0, 1]).all()

    assert events[
        "failure_flag"
    ].isin([0, 1]).all()

    assert events[
        "failure_severity"
    ].isin(
        [
            "None",
            "Minor",
            "Moderate",
            "Major",
            "Critical",
        ]
    ).all()

    assert events[
        "utilization_rate"
    ].between(
        0,
        1,
    ).all()

    assert events[
        "degradation_index"
    ].between(
        0,
        1,
    ).all()