import pandas as pd

from src.data.generators.project_generator import (
    generate_projects,
)

from src.data.generators.activity_generator import (
    generate_project_activities,
)


def test_activity_generation():

    projects = generate_projects(
        n=5,
        seed=42,
    )

    activities = (
        generate_project_activities(
            projects,
            seed=42,
        )
    )

    assert isinstance(
        activities,
        pd.DataFrame,
    )

    assert len(activities) > 0

    assert activities[
        "activity_id"
    ].is_unique

    assert (
        activities["project_id"]
        .isin(projects["project_id"])
        .all()
    )

    assert (
        activities[
            "planned_duration_days"
        ] > 0
    ).all()

    assert activities[
        "progress_pct"
    ].between(0, 100).all()

    assert activities[
        "critical_path_flag"
    ].isin([0, 1]).all()

    assert activities[
        "delay_flag"
    ].isin([0, 1]).all()