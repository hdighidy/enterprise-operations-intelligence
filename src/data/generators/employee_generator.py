from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker


DEPARTMENTS = [
    "Project Management",
    "Procurement",
    "Finance",
    "Engineering",
    "Quality",
    "Construction",
    "Supply Chain",
    "IT",
]


ROLES = [
    "Engineer",
    "Project Manager",
    "Procurement Specialist",
    "Planner",
    "Quantity Surveyor",
    "Data Analyst",
    "Quality Engineer",
    "Supervisor",
    "Manager",
]


def generate_employees(
    n: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    fake = Faker()
    Faker.seed(seed)

    records = []

    for i in range(1, n + 1):

        records.append(
            {
                "employee_id": f"EMP-{i:05d}",
                "employee_name": fake.name(),
                "department": rng.choice(DEPARTMENTS),
                "role": rng.choice(ROLES),
                "experience_years": int(
                    rng.integers(1, 31)
                ),
            }
        )

    return pd.DataFrame(records)