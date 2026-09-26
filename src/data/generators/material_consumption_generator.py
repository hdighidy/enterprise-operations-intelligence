from __future__ import annotations

import numpy as np
import pandas as pd


def generate_material_consumption(
    projects: pd.DataFrame,
    materials: pd.DataFrame,
    activities: pd.DataFrame,
    deliveries: pd.DataFrame,
    seed: int = 42,
    max_activities: int = 3000,
    materials_per_activity: int = 2,
) -> pd.DataFrame:
    """
    Generate development-scale material consumption data.

    Designed for local development on resource-constrained machines.

    Default target:
        ~150K-250K records depending on activity durations.

    Parameters
    ----------
    max_activities:
        Maximum number of activities used to generate consumption.

    materials_per_activity:
        Number of materials associated with each activity.

    The generator preserves relationships between:
        Project
        Activity
        Material
        Delivery
        Consumption
    """

    rng = np.random.default_rng(seed)

    # ==========================================================
    # Validate input columns
    # ==========================================================

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

    # ==========================================================
    # Copy data
    # ==========================================================

    projects = projects.copy()
    materials = materials.copy()
    activities = activities.copy()
    deliveries = deliveries.copy()

    activities["planned_start_date"] = pd.to_datetime(
        activities["planned_start_date"]
    )

    activities["planned_finish_date"] = pd.to_datetime(
        activities["planned_finish_date"]
    )

    deliveries["actual_delivery_date"] = pd.to_datetime(
        deliveries["actual_delivery_date"]
    )

    # ==========================================================
    # Limit activities for local development
    # ==========================================================

    if len(activities) > max_activities:

        activities = activities.sample(
            n=max_activities,
            random_state=seed,
        )

        activities = activities.sort_values(
            [
                "project_id",
                "planned_start_date",
            ]
        )

        activities = activities.reset_index(
            drop=True
        )

    # ==========================================================
    # Lookup dictionaries
    # ==========================================================

    project_lookup = (
        projects
        .set_index("project_id")
        .to_dict("index")
    )

    # ==========================================================
    # Pre-aggregate deliveries
    #
    # This is the major performance improvement.
    #
    # Instead of filtering the entire deliveries dataframe
    # for every consumption record, we aggregate once.
    # ==========================================================

    delivery_summary = (
        deliveries[
            [
                "project_id",
                "material_id",
                "actual_delivery_date",
                "delivered_quantity",
            ]
        ]
        .sort_values(
            [
                "project_id",
                "material_id",
                "actual_delivery_date",
            ]
        )
        .copy()
    )

    delivery_summary[
        "cumulative_delivered_quantity"
    ] = (
        delivery_summary
        .groupby(
            [
                "project_id",
                "material_id",
            ]
        )["delivered_quantity"]
        .cumsum()
    )

    # ==========================================================
    # Create lookup structure for delivery history
    # ==========================================================

    delivery_lookup = {}

    for (
        project_id,
        material_id,
    ), group in delivery_summary.groupby(
        [
            "project_id",
            "material_id",
        ]
    ):

        delivery_lookup[
            (
                project_id,
                material_id,
            )
        ] = (
            group[
                [
                    "actual_delivery_date",
                    "cumulative_delivered_quantity",
                ]
            ]
            .sort_values(
                "actual_delivery_date"
            )
            .reset_index(drop=True)
        )

    # ==========================================================
    # Prepare materials
    # ==========================================================

    material_records = materials[
        [
            "material_id",
            "criticality",
            "standard_price",
        ]
    ].to_dict("records")

    records = []

    consumption_id = 1

    # ==========================================================
    # Generate consumption
    # ==========================================================

    for activity_index, (_, activity) in enumerate(
        activities.iterrows(),
        start=1,
    ):

        project_id = activity["project_id"]

        if project_id not in project_lookup:
            continue

        project = project_lookup[
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
        }.get(
            complexity,
            1.0,
        )

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

        # ------------------------------------------------------
        # Select materials
        # ------------------------------------------------------

        selected_materials = rng.choice(
            material_records,
            size=min(
                materials_per_activity,
                len(material_records),
            ),
            replace=False,
        )

        for material in selected_materials:

            material_id = material[
                "material_id"
            ]

            criticality = material[
                "criticality"
            ]

            criticality_factor = {
                "LOW": 0.80,
                "MEDIUM": 1.00,
                "HIGH": 1.30,
                "CRITICAL": 1.60,
            }.get(
                criticality,
                1.0,
            )

            # --------------------------------------------------
            # Base consumption rate
            # --------------------------------------------------

            base_daily_consumption = rng.uniform(
                3,
                20,
            )

            planned_consumption = (
                base_daily_consumption
                * complexity_factor
                * criticality_factor
            )

            planned_consumption = round(
                max(
                    0.5,
                    planned_consumption,
                ),
                2,
            )

            # --------------------------------------------------
            # Activity progress
            # --------------------------------------------------

            progress_factor = max(
                0.15,
                min(
                    1.0,
                    float(
                        activity[
                            "progress_pct"
                        ]
                    )
                    / 100.0,
                ),
            )

            actual_consumption = (
                planned_consumption
                * progress_factor
            )

            # --------------------------------------------------
            # Delivery lookup
            # --------------------------------------------------

            delivery_history = delivery_lookup.get(
                (
                    project_id,
                    material_id,
                )
            )

            # --------------------------------------------------
            # Daily records
            # --------------------------------------------------

            for day in range(
                duration + 1
            ):

                consumption_date = (
                    start_date
                    + pd.Timedelta(
                        days=day
                    )
                )

                # Weekend reduction
                weekday_factor = (
                    0.90
                    if consumption_date.weekday()
                    >= 5
                    else 1.0
                )

                daily_planned = (
                    planned_consumption
                    * weekday_factor
                )

                daily_actual = (
                    actual_consumption
                    * weekday_factor
                    * rng.uniform(
                        0.85,
                        1.15,
                    )
                )

                daily_planned = round(
                    max(
                        0,
                        daily_planned,
                    ),
                    2,
                )

                daily_actual = round(
                    max(
                        0,
                        daily_actual,
                    ),
                    2,
                )

                # --------------------------------------------------
                # Determine available delivered quantity
                #
                # Search only the delivery history for this
                # project/material combination.
                # --------------------------------------------------

                if delivery_history is None:

                    available_quantity = 0.0

                else:

                    dates = (
                        delivery_history[
                            "actual_delivery_date"
                        ].values
                    )

                    position = np.searchsorted(
                        dates,
                        np.datetime64(
                            consumption_date
                        ),
                        side="right",
                    )

                    if position == 0:

                        available_quantity = 0.0

                    else:

                        available_quantity = float(
                            delivery_history.iloc[
                                position - 1
                            ][
                                "cumulative_delivered_quantity"
                            ]
                        )

                # --------------------------------------------------
                # Consumption-to-availability signal
                # --------------------------------------------------

                stockout_flag = int(
                    available_quantity
                    < daily_actual
                )

                utilization_ratio = (
                    daily_actual
                    / max(
                        available_quantity,
                        1.0,
                    )
                )

                utilization_ratio = min(
                    utilization_ratio,
                    10.0,
                )

                # --------------------------------------------------
                # Record
                # --------------------------------------------------

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
                                available_quantity,
                                2,
                            )
                        ),
                        "utilization_ratio": round(
                            utilization_ratio,
                            4,
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

        # ------------------------------------------------------
        # Progress message
        # ------------------------------------------------------

        if (
            activity_index % 500 == 0
        ):

            print(
                "Material consumption progress: "
                f"{activity_index:,}/"
                f"{len(activities):,} activities"
            )

    return pd.DataFrame(records)