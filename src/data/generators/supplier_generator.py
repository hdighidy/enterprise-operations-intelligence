from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker


SUPPLIER_CATEGORIES = [
    "HVAC",
    "Electrical",
    "Fire Fighting",
    "Plumbing",
    "Civil",
    "General Materials",
]


def generate_suppliers(
    n: int = 350,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic supplier master data.

    Supplier reliability is intentionally heterogeneous
    because later ML models need meaningful behavioral patterns.
    """

    rng = np.random.default_rng(seed)

    fake = Faker()
    Faker.seed(seed)

    records = []

    for i in range(1, n + 1):

        supplier_id = f"SUP-{i:04d}"

        reliability = float(
            rng.beta(8, 2)
        )

        capacity = int(
            rng.integers(500, 20_000)
        )

        supplier_rating = round(
            1 + reliability * 4,
            2,
        )

        records.append(
            {
                "supplier_id": supplier_id,
                "supplier_name": fake.company(),
                "supplier_category": rng.choice(
                    SUPPLIER_CATEGORIES
                ),
                "location": fake.city(),
                "supplier_rating": supplier_rating,
                "capacity": capacity,
                "payment_terms_days": int(
                    rng.choice([30, 45, 60, 90])
                ),
                "reliability_score": round(
                    reliability,
                    4,
                ),
            }
        )

    return pd.DataFrame(records)