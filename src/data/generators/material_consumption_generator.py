from __future__ import annotations

import numpy as np
import pandas as pd


def generate_material_consumption(
    projects: pd.DataFrame,
    materials: pd.DataFrame,
    activities: pd.DataFrame,
    deliveries: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate daily material consumption records.

    Consumption is influenced by:
        - project activity
        - project complexity
        - material criticality
        - project progress
        - delivered quantities

    This dataset will later support:
        - demand forecasting
        - stockout prediction
        - inventory optimization
        - material planning
    """

    rng = np.random.default_rng(seed)

    required_project_columns = {
        "project_id",
        "project_complexity",
    }

    required_material_columns = {
        "material_id",
        "criticality",
        "standard_price",
    }

    required_activity_columns = {
        "activity_id",
        "project_id",
        "planned_start_date",
        "planned_finish_date",
        "progress_pct",
    }

    required_delivery_columns = {
        "project_id",
        "material_id",
        "delivered_quantity",
        "actual_delivery_date",
    }

    missing = (
        required_project_columns
        - set(projects.columns)
    )

    if missing:
        raise ValueError(
            f"Missing project columns: {sorted(missing)}"
        )

    missing = (
        required_material_columns
        - set(materials.columns)
    )

    if missing:
        raise ValueError(
            f"Missing material columns: {sorted(missing)}"
        )

    missing = (
        required_activity_columns
        - set(activities.columns)
    )

    if missing:
        raise ValueError(
            f"Missing activity columns: {sorted(missing)}"
        )

    missing = (
        required_delivery_columns
        - set(deliveries.columns)
    )

    if missing:
        raise ValueError(
            f"Missing delivery columns: {sorted(missing)}"
        )

    projects = projects.copy()
    materials = materials.copy()
    activities = activities.copy()
    deliveries = deliveries.copy()

    activities["planned_start_date"] = pd.to_datetime(
        activities["planned_start_date"])

    activities["planned_finish_date"] = pd.to_datetime(
        activities["planned_finish_date"])

    deliveries["actual_delivery_date"] = pd.to_datetime(
        deliveries["actual_delivery_date"])

    project_lookup = projects.set_index(
        "project_id")

    material_lookup = materials.set_index(
        "material_id")

    records = []

    consumption_id = 1

    # ---------------------------------------------------------
    # Generate consumption around project activities
    # ---------------------------------------------------------

    for _, activity in activities.iterrows():

        project_id = activity["project_id"]

        if project_id not in project_lookup.index:
            continue

        project = project_lookup.loc[
            project_id
        ]

        complexity = project[
            "project_complexity"
        ]

        complexity_factor = {
            "LOW": 0.80,
            "MEDIUM": 1.00,
            "HIGH": 1.30,
            "VERY_HIGH": 1.60,
        }.get(complexity, 1.0)

        start_date = pd.Timestamp(
            activity["planned_start_date"]
        )

        finish_date = pd.Timestamp(
            activity["planned_finish_date"]
        )

        duration = max(
            1,
            (finish_date - start_date).days,
        )

        # Not every material is consumed by every activity.
        material_sample_size = min(
            8,
            len(materials),
        )

        selected_materials = materials.sample(
            n=material_sample_size,
            random_state=int(
                rng.integers(
                    0,
                    1_000_000,
                )
            ),
        )

        for _, material in selected_materials.iterrows():

            material_id = material["material_id"]

            criticality = material[
                "criticality"
            ]

            criticality_factor = {
                "LOW": 0.8,
                "MEDIUM": 1.0,
                "HIGH": 1.3,
                "CRITICAL": 1.6,
            }.get(criticality, 1.0)

            # -------------------------------------------------
            # Daily consumption
            # -------------------------------------------------

            base_daily_consumption = rng.uniform(
                2,
                30,
            )

            planned_consumption = (
                base_daily_consumption
                * complexity_factor
                * criticality_factor
            )

            planned_consumption = max(
                0.5,
                planned_consumption,
            )

            planned_consumption = round(
                planned_consumption,
                2,
            )

            # -------------------------------------------------
            # Actual consumption
            # -------------------------------------------------

            progress_factor = max(
                0.1,
                float(activity["progress_pct"])
                / 100.0,
            )

            actual_consumption = (
                planned_consumption
                * progress_factor
                * rng.uniform(
                    0.75,
                    1.25,
                )
            )

            actual_consumption = max(
                0,
                actual_consumption,
            )

            actual_consumption = round(
                actual_consumption,
                2,
            )

            # -------------------------------------------------
            # Daily records
            # -------------------------------------------------

            for day in range(duration + 1):

                consumption_date = (
                    start_date
                    + pd.Timedelta(
                        days=day
                    )
                )

                # Small daily seasonality
                weekday_factor = (
                    0.90
                    if consumption_date.weekday()
                    >= 5
                    else 1.0
                )

                daily_planned = round(
                    planned_consumption
                    * weekday_factor,
                    2,
                )

                daily_actual = round(
                    actual_consumption
                    * weekday_factor
                    * rng.uniform(
                        0.85,
                        1.15,
                    ),
                    2,
                )

                # -------------------------------------------------
                # Delivery availability
                # -------------------------------------------------

                delivered_before_date = deliveries[
                    (
                        deliveries["project_id"]
                        == project_id
                    )
                    &
                    (
                        deliveries["material_id"]
                        == material_id
                    )
                    &
                    (
                        deliveries[
                            "actual_delivery_date"
                        ]
                        <= consumption_date
                    )
                ]

                available_quantity = (
                    delivered_before_date[
                        "delivered_quantity"
                    ].sum()
                )

                # -------------------------------------------------
                # Inventory signal
                # -------------------------------------------------

                cumulative_consumption = (
                    daily_actual
                )

                stockout_flag = int(
                    available_quantity
                    < cumulative_consumption
                )

                # -------------------------------------------------
                # Record
                # -------------------------------------------------

                records.append(
                    {
                        "consumption_id": (
                            f"CONS-{consumption_id:08d}"
                        ),
                        "date": (
                            consumption_date.date()
                        ),
                        "project_id": project_id,
                        "activity_id": activity[
                            "activity_id"
                        ],
                        "material_id": material_id,
                        "planned_daily_quantity": (
                            daily_planned
                        ),
                        "actual_daily_quantity": (
                            daily_actual
                        ),
                        "available_delivered_quantity": (
                            round(
                                float(
                                    available_quantity
                                ),
                                2,
                            )
                        ),
                        "stockout_flag": (
                            stockout_flag
                        ),
                        "material_unit_price": (
                            float(
                                material[
                                    "standard_price"
                                ]
                            )
                        ),
                    }
                )

                consumption_id += 1

    return pd.DataFrame(records)