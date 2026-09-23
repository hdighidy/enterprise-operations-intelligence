from __future__ import annotations

import numpy as np
import pandas as pd


EQUIPMENT_TYPES = [
    "Crane",
    "Excavator",
    "Forklift",
    "Generator",
    "Compressor",
    "Concrete Pump",
    "Welding Machine",
    "Manlift",
]


def generate_equipment(
    n: int = 1_000,
    seed: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    records = []

    for i in range(1, n + 1):

        operating_hours = int(
            rng.integers(100, 20_000)
        )

        downtime_hours = round(
            float(
                rng.gamma(
                    shape=2,
                    scale=20,
                )
            ),
            2,
        )

        total_hours = operating_hours + downtime_hours

        utilization_rate = (
            operating_hours / total_hours
        )

        records.append(
            {
                "equipment_id": f"EQP-{i:05d}",
                "equipment_type": rng.choice(
                    EQUIPMENT_TYPES
                ),
                "operating_hours": operating_hours,
                "downtime_hours": downtime_hours,
                "utilization_rate": round(
                    utilization_rate,
                    4,
                ),
                "maintenance_cost": round(
                    float(
                        rng.lognormal(
                            mean=8,
                            sigma=1.2,
                        )
                    ),
                    2,
                ),
            }
        )

    return pd.DataFrame(records)