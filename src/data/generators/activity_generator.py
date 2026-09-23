from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


ACTIVITY_TEMPLATES = [
    ("Mobilization", "General"),
    ("Site Preparation", "Civil"),
    ("Excavation", "Civil"),
    ("Foundation Works", "Civil"),
    ("Concrete Works", "Structural"),
    ("Structural Steel", "Structural"),
    ("Block Works", "Architectural"),
    ("Waterproofing", "Architectural"),
    ("Plastering", "Architectural"),
    ("Floor Finishing", "Architectural"),
    ("Painting", "Architectural"),
    ("HVAC Installation", "MEP"),
    ("Electrical Installation", "MEP"),
    ("Plumbing Installation", "MEP"),
    ("Fire Fighting Installation", "MEP"),
    ("Fire Alarm Installation", "MEP"),
    ("Testing & Commissioning", "Commissioning"),
    ("Final Inspection", "Commissioning"),
    ("Handover", "General"),
]


ACTIVITY_WEIGHTS = {
    "LOW": 1.0,
    "MEDIUM": 1.5,
    "HIGH": 2.0,
    "VERY_HIGH": 2.8,
}


def generate_project_activities(
    projects: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate schedule activities for every project.

    Activity duration, cost and delay behavior are influenced
    by project complexity.
    """

    rng = np.random.default_rng(seed)

    records = []

    activity_counter = 1

    for _, project in projects.iterrows():

        complexity = project["project_complexity"]

        complexity_factor = ACTIVITY_WEIGHTS[
            complexity
        ]

        project_start = pd.Timestamp(
            project["planned_start_date"]
        )

        project_end = pd.Timestamp(
            project["planned_end_date"]
        )

        project_duration = (
            project_end - project_start
        ).days

        # Number of activities grows with complexity
        base_activity_count = int(
            rng.integers(40, 90)
        )

        number_of_activities = int(
            base_activity_count
            * complexity_factor
        )

        number_of_activities = min(
            number_of_activities,
            180,
        )

        number_of_activities = max(
            number_of_activities,
            40,
        )

        # Generate activity positions
        start_offsets = np.sort(
            rng.integers(
                0,
                max(
                    1,
                    project_duration - 30,
                ),
                size=number_of_activities,
            )
        )

        for sequence in range(
            number_of_activities
        ):

            template_name, activity_type = (
                ACTIVITY_TEMPLATES[
                    sequence
                    % len(ACTIVITY_TEMPLATES)
                ]
            )

            planned_duration = int(
                rng.integers(
                    5,
                    max(
                        6,
                        int(
                            60
                            * complexity_factor
                        ),
                    ),
                )
            )

            planned_start = (
                project_start
                + timedelta(
                    days=int(
                        start_offsets[
                            sequence
                        ]
                    )
                )
            )

            planned_finish = (
                planned_start
                + timedelta(
                    days=planned_duration
                )
            )

            # Keep activities inside the project window
            if planned_finish > project_end:

                planned_finish = project_end

                planned_duration = max(
                    1,
                    (
                        planned_finish
                        - planned_start
                    ).days,
                )

            # Critical path probability
            critical_probability = {
                "LOW": 0.15,
                "MEDIUM": 0.25,
                "HIGH": 0.35,
                "VERY_HIGH": 0.45,
            }[complexity]

            critical_path_flag = int(
                rng.random()
                < critical_probability
            )

            # Activity cost
            base_cost = (
                project["contract_value"]
                / number_of_activities
            )

            baseline_cost = (
                base_cost
                * rng.uniform(
                    0.5,
                    1.8,
                )
            )

            baseline_cost = round(
                float(baseline_cost),
                2,
            )

            # Progress
            current_date = pd.Timestamp(
                "2026-09-23"
            )

            if current_date < planned_start:

                progress = 0.0

            elif current_date >= planned_finish:

                progress = 100.0

            else:

                elapsed = (
                    current_date
                    - planned_start
                ).days

                progress = min(
                    100.0,
                    max(
                        0.0,
                        (
                            elapsed
                            / planned_duration
                        )
                        * 100,
                    ),
                )

                # Add realistic execution variability
                progress += rng.normal(
                    0,
                    8,
                )

                progress = min(
                    100.0,
                    max(
                        0.0,
                        progress,
                    ),
                )

            # Activity delay risk
            delay_probability = (
                0.05
                + (
                    0.12
                    if critical_path_flag
                    else 0
                )
                + (
                    0.08
                    if complexity
                    in [
                        "HIGH",
                        "VERY_HIGH",
                    ]
                    else 0
                )
            )

            activity_delayed = (
                rng.random()
                < delay_probability
            )

            if activity_delayed:

                delay_days = int(
                    rng.gamma(
                        shape=2.0,
                        scale=8.0,
                    )
                )

                delay_days = max(
                    2,
                    delay_days,
                )

            else:

                delay_days = 0

            actual_finish = (
                planned_finish
                + timedelta(
                    days=delay_days
                )
            )

            actual_cost = (
                baseline_cost
                * rng.normal(
                    1.0
                    + (
                        0.002
                        * delay_days
                    ),
                    0.08,
                )
            )

            actual_cost = max(
                0,
                float(actual_cost),
            )

            records.append(
                {
                    "activity_id": (
                        f"ACT-{activity_counter:06d}"
                    ),
                    "project_id": project[
                        "project_id"
                    ],
                    "activity_sequence": (
                        sequence + 1
                    ),
                    "activity_name": (
                        template_name
                    ),
                    "activity_type": (
                        activity_type
                    ),
                    "planned_start_date": (
                        planned_start.date()
                    ),
                    "planned_finish_date": (
                        planned_finish.date()
                    ),
                    "planned_duration_days": (
                        planned_duration
                    ),
                    "actual_finish_date": (
                        actual_finish.date()
                    ),
                    "progress_pct": round(
                        progress,
                        2,
                    ),
                    "critical_path_flag": (
                        critical_path_flag
                    ),
                    "baseline_cost": (
                        baseline_cost
                    ),
                    "actual_cost": round(
                        actual_cost,
                        2,
                    ),
                    "delay_days": delay_days,
                    "delay_flag": int(
                        delay_days > 7
                    ),
                }
            )

            activity_counter += 1

    return pd.DataFrame(records)