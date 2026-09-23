from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


PROJECT_TYPES = [
    "Commercial",
    "Residential",
    "Industrial",
    "Hospital",
    "Infrastructure",
    "Mixed Use",
    "Educational",
]

LOCATIONS = [
    "Cairo",
    "Alexandria",
    "Giza",
    "New Cairo",
    "New Capital",
    "6th of October",
    "North Coast",
    "Ain Sokhna",
    "Hurghada",
]

CLIENT_TYPES = [
    "Government",
    "Real Estate Developer",
    "Industrial Company",
    "Private Investor",
    "International Contractor",
]

PROJECT_STATUSES = [
    "PLANNING",
    "ACTIVE",
    "COMPLETED",
]


def generate_projects(
    n: int = 150,
    seed: int = 42,
    employee_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic enterprise project master data.

    Parameters
    ----------
    n:
        Number of projects.

    seed:
        Random seed for reproducibility.

    employee_ids:
        Optional list of employee IDs that can be assigned
        as project managers.

    Returns
    -------
    pandas.DataFrame
        Project master dataset.
    """

    rng = np.random.default_rng(seed)

    fake = Faker()
    Faker.seed(seed)

    if employee_ids is None:
        employee_ids = [
            f"EMP-{i:05d}"
            for i in range(1, 201)
        ]

    records = []

    start_date = datetime(2017, 1, 1)
    end_date = datetime(2026, 6, 30)

    total_days = (
        end_date - start_date
    ).days

    for i in range(1, n + 1):

        project_type = rng.choice(
            PROJECT_TYPES
        )

        location = rng.choice(
            LOCATIONS
        )

        client_type = rng.choice(
            CLIENT_TYPES
        )

        # ----------------------------------------------------
        # Project complexity
        # ----------------------------------------------------

        complexity = rng.choice(
            ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
            p=[0.20, 0.40, 0.30, 0.10],
        )

        complexity_factor = {
            "LOW": 0.8,
            "MEDIUM": 1.0,
            "HIGH": 1.4,
            "VERY_HIGH": 1.9,
        }[complexity]

        # ----------------------------------------------------
        # Contract value
        # ----------------------------------------------------

        base_value = rng.lognormal(
            mean=17.0,
            sigma=1.0,
        )

        contract_value = (
            base_value * complexity_factor
        )

        contract_value = round(
            float(contract_value),
            2,
        )

        # ----------------------------------------------------
        # Planned duration
        # ----------------------------------------------------

        base_duration = rng.integers(
            180,
            1_000,
        )

        planned_duration = int(
            base_duration * complexity_factor
        )

        # ----------------------------------------------------
        # Planned start
        # ----------------------------------------------------

        random_offset = int(
            rng.integers(
                0,
                total_days,
            )
        )

        planned_start = (
            start_date
            + timedelta(
                days=random_offset
            )
        )

        planned_end = (
            planned_start
            + timedelta(
                days=planned_duration
            )
        )

        # ----------------------------------------------------
        # Actual start
        # ----------------------------------------------------

        start_variance = int(
            rng.normal(
                loc=5,
                scale=15,
            )
        )

        actual_start = (
            planned_start
            + timedelta(
                days=start_variance
            )
        )

        # ----------------------------------------------------
        # Project delay behavior
        # ----------------------------------------------------

        risk_probability = {
            "LOW": 0.12,
            "MEDIUM": 0.22,
            "HIGH": 0.35,
            "VERY_HIGH": 0.48,
        }[complexity]

        delayed = (
            rng.random()
            < risk_probability
        )

        if delayed:

            delay_days = int(
                rng.gamma(
                    shape=2.0,
                    scale=25.0,
                )
            )

            delay_days = max(
                5,
                delay_days,
            )

        else:

            delay_days = int(
                rng.integers(
                    -5,
                    6,
                )
            )

        actual_end = (
            planned_end
            + timedelta(
                days=delay_days
            )
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if actual_end < datetime(2025, 1, 1):

            status = "COMPLETED"

        elif actual_start <= datetime.now():

            status = "ACTIVE"

        else:

            status = "PLANNING"

        # ----------------------------------------------------
        # Project manager
        # ----------------------------------------------------

        project_manager_id = rng.choice(
            employee_ids
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        records.append(
            {
                "project_id": f"PRJ-{i:04d}",
                "project_name": (
                    f"{project_type} "
                    f"Development {i:03d}"
                ),
                "project_type": project_type,
                "location": location,
                "client_type": client_type,
                "contract_value": contract_value,
                "planned_start_date": (
                    planned_start.date()
                ),
                "planned_end_date": (
                    planned_end.date()
                ),
                "actual_start_date": (
                    actual_start.date()
                ),
                "actual_end_date": (
                    actual_end.date()
                ),
                "planned_duration_days": (
                    planned_duration
                ),
                "actual_duration_days": (
                    (
                        actual_end
                        - actual_start
                    ).days
                ),
                "project_complexity": complexity,
                "project_manager_id": (
                    project_manager_id
                ),
                "project_status": status,
                "delay_days": delay_days,
                "delay_flag": int(
                    delay_days > 14
                ),
            }
        )

    return pd.DataFrame(records)