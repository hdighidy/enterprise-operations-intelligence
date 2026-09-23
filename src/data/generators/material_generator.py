from __future__ import annotations

import numpy as np
import pandas as pd


MATERIAL_CATEGORIES = {
    "HVAC": [
        "Air Handling Unit",
        "Fan Coil Unit",
        "Chiller",
        "Cooling Tower",
        "Ventilation Fan",
    ],
    "Electrical": [
        "Distribution Board",
        "Cable",
        "Circuit Breaker",
        "Transformer",
        "Lighting Fixture",
    ],
    "Fire Fighting": [
        "Fire Fighting Pump",
        "Fire Hose Reel",
        "Fire Extinguisher",
        "Sprinkler",
        "Fire Alarm Panel",
    ],
    "Plumbing": [
        "Water Pump",
        "Pipe",
        "Valve",
        "Water Tank",
        "Drainage Fixture",
    ],
    "Civil": [
        "Concrete",
        "Rebar",
        "Structural Steel",
        "Blocks",
        "Waterproofing Material",
    ],
}


def generate_materials(
    n: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic enterprise material master data.

    Parameters
    ----------
    n:
        Number of materials to generate.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Material master dataset.
    """

    rng = np.random.default_rng(seed)

    categories = list(MATERIAL_CATEGORIES.keys())

    records = []

    for i in range(1, n + 1):

        category = rng.choice(categories)

        material_name = rng.choice(
            MATERIAL_CATEGORIES[category]
        )

        material_id = f"MAT-{i:04d}"

        criticality = rng.choice(
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            p=[0.30, 0.35, 0.25, 0.10],
        )

        unit = rng.choice(
            ["UNIT", "METER", "KG", "TON", "SET"]
        )

        standard_price = round(
            float(rng.lognormal(mean=8.5, sigma=1.0)),
            2,
        )

        lead_time_days = int(
            rng.integers(3, 90)
        )

        minimum_order_quantity = int(
            rng.integers(1, 100)
        )

        records.append(
            {
                "material_id": material_id,
                "material_code": f"MAT-CODE-{i:05d}",
                "material_name": material_name,
                "category": category,
                "unit": unit,
                "standard_price": standard_price,
                "criticality": criticality,
                "lead_time_days": lead_time_days,
                "minimum_order_quantity": minimum_order_quantity,
            }
        )

    return pd.DataFrame(records)