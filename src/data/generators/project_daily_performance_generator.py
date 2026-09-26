from __future__ import annotations

import numpy as np
import pandas as pd


def generate_project_daily_performance(
    projects: pd.DataFrame,
    activities: pd.DataFrame,
    deliveries: pd.DataFrame,
    material_consumption: pd.DataFrame,
    equipment_events: pd.DataFrame,
    seed: int = 42,
    max_projects: int = 150,
) -> pd.DataFrame:
    """
    Generate daily project-level operational performance.

    Development target:
        approximately 50K-150K records depending on
        project duration.

    The dataset combines:
        - project schedule
        - activity progress
        - material availability
        - supplier/delivery delays
        - equipment downtime
        - equipment failures
        - project cost signals

    Important:
        Final project delay_flag is NOT used as a predictor.
        It is retained only as a future supervised-learning target.
    """

    rng = np.random.default_rng(seed)

    # ==========================================================
    # Validate inputs
    # ==========================================================

    required_project_columns = {
        "project_id",
        "planned_start_date",
        "planned_end_date",
        "actual_start_date",
        "actual_end_date",
        "planned_duration_days",
        "actual_duration_days",
        "contract_value",
        "project_complexity",
        "delay_days",
        "delay_flag",
    }

    required_activity_columns = {
        "activity_id",
        "project_id",
        "planned_start_date",
        "planned_finish_date",
        "progress_pct",
        "critical_path_flag",
        "delay_days",
        "delay_flag",
        "baseline_cost",
        "actual_cost",
    }

    required_delivery_columns = {
        "project_id",
        "material_id",
        "actual_delivery_date",
        "delivered_quantity",
    }

    required_consumption_columns = {
        "date",
        "project_id",
        "material_id",
        "planned_daily_quantity",
        "actual_daily_quantity",
        "available_delivered_quantity",
        "stockout_flag",
    }

    required_equipment_columns = {
        "event_date",
        "equipment_id",
        "operating_hours",
        "downtime_hours",
        "maintenance_flag",
        "failure_flag",
        "maintenance_cost",
    }

    datasets = [
        (
            projects,
            required_project_columns,
            "projects",
        ),
        (
            activities,
            required_activity_columns,
            "activities",
        ),
        (
            deliveries,
            required_delivery_columns,
            "deliveries",
        ),
        (
            material_consumption,
            required_consumption_columns,
            "material_consumption",
        ),
        (
            equipment_events,
            required_equipment_columns,
            "equipment_events",
        ),
    ]

    for dataframe, required, name in datasets:

        missing = required - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"Missing {name} columns: "
                f"{sorted(missing)}"
            )

    # ==========================================================
    # Copy and normalize dates
    # ==========================================================

    projects = projects.copy()
    activities = activities.copy()
    deliveries = deliveries.copy()
    material_consumption = material_consumption.copy()
    equipment_events = equipment_events.copy()

    date_columns = {
        "projects": [
            "planned_start_date",
            "planned_end_date",
            "actual_start_date",
            "actual_end_date",
        ],
        "activities": [
            "planned_start_date",
            "planned_finish_date",
        ],
        "deliveries": [
            "actual_delivery_date",
        ],
        "material_consumption": [
            "date",
        ],
        "equipment_events": [
            "event_date",
        ],
    }

    for column in date_columns["projects"]:
        projects[column] = pd.to_datetime(
            projects[column]
        )

    for column in date_columns["activities"]:
        activities[column] = pd.to_datetime(
            activities[column]
        )

    for column in date_columns["deliveries"]:
        deliveries[column] = pd.to_datetime(
            deliveries[column]
        )

    material_consumption["date"] = pd.to_datetime(
        material_consumption["date"]
    )

    equipment_events["event_date"] = pd.to_datetime(
        equipment_events["event_date"]
    )

    # ==========================================================
    # Limit projects for local development
    # ==========================================================

    if len(projects) > max_projects:

        projects = projects.sample(
            n=max_projects,
            random_state=seed,
        )

    projects = projects.reset_index(
        drop=True
    )

    valid_project_ids = set(
        projects["project_id"]
    )

    activities = activities[
        activities["project_id"].isin(
            valid_project_ids
        )
    ].copy()

    deliveries = deliveries[
        deliveries["project_id"].isin(
            valid_project_ids
        )
    ].copy()

    material_consumption = material_consumption[
        material_consumption["project_id"].isin(
            valid_project_ids
        )
    ].copy()

    # ==========================================================
    # Pre-aggregate activity data
    # ==========================================================

    activity_daily = []

    for _, activity in activities.iterrows():

        start = activity[
            "planned_start_date"
        ]

        finish = activity[
            "planned_finish_date"
        ]

        duration = max(
            1,
            (finish - start).days,
        )

        for day in range(duration + 1):

            date = (
                start
                + pd.Timedelta(days=day)
            )

            activity_daily.append(
                {
                    "project_id": activity[
                        "project_id"
                    ],
                    "date": date,
                    "activity_id": activity[
                        "activity_id"
                    ],
                    "critical_path_flag": int(
                        activity[
                            "critical_path_flag"
                        ]
                    ),
                    "activity_delay_days": float(
                        activity[
                            "delay_days"
                        ]
                    ),
                    "activity_delay_flag": int(
                        activity[
                            "delay_flag"
                        ]
                    ),
                    "baseline_cost": float(
                        activity[
                            "baseline_cost"
                        ]
                    ),
                    "actual_cost": float(
                        activity[
                            "actual_cost"
                        ]
                    ),
                }
            )

    activity_daily = pd.DataFrame(
        activity_daily
    )

    if activity_daily.empty:

        raise ValueError(
            "No activity records available."
        )

    # Aggregate daily activity signals

    activity_daily = (
        activity_daily
        .groupby(
            [
                "project_id",
                "date",
            ]
        )
        .agg(
            active_activity_count=(
                "activity_id",
                "nunique",
            ),
            critical_activity_count=(
                "critical_path_flag",
                "sum",
            ),
            delayed_activity_count=(
                "activity_delay_flag",
                "sum",
            ),
            activity_delay_days=(
                "activity_delay_days",
                "sum",
            ),
            baseline_activity_cost=(
                "baseline_cost",
                "sum",
            ),
            actual_activity_cost=(
                "actual_cost",
                "sum",
            ),
        )
        .reset_index()
    )

    # ==========================================================
    # Material consumption aggregation
    # ==========================================================

    material_daily = (
        material_consumption
        .groupby(
            [
                "project_id",
                "date",
            ]
        )
        .agg(
            planned_material_quantity=(
                "planned_daily_quantity",
                "sum",
            ),
            actual_material_quantity=(
                "actual_daily_quantity",
                "sum",
            ),
            material_stockout_count=(
                "stockout_flag",
                "sum",
            ),
            material_records=(
                "material_id",
                "count",
            ),
        )
        .reset_index()
    )

    # ==========================================================
    # Delivery aggregation
    # ==========================================================

    deliveries["delivery_date"] = (
        deliveries[
            "actual_delivery_date"
        ].dt.normalize()
    )

    delivery_daily = (
        deliveries
        .groupby(
            [
                "project_id",
                "delivery_date",
            ]
        )
        .agg(
            delivery_count=(
                "material_id",
                "count",
            ),
            delivered_quantity=(
                "delivered_quantity",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "delivery_date": "date"
            }
        )
    )

    # ==========================================================
    # Equipment aggregation
    #
    # Equipment events don't currently contain project_id,
    # so they are treated as an enterprise operational signal.
    #
    # This will later be connected to projects through an
    # equipment assignment table.
    # ==========================================================

    equipment_daily = (
        equipment_events
        .groupby(
            "event_date"
        )
        .agg(
            equipment_operating_hours=(
                "operating_hours",
                "sum",
            ),
            equipment_downtime_hours=(
                "downtime_hours",
                "sum",
            ),
            equipment_failure_count=(
                "failure_flag",
                "sum",
            ),
            equipment_maintenance_count=(
                "maintenance_flag",
                "sum",
            ),
            equipment_maintenance_cost=(
                "maintenance_cost",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "event_date": "date"
            }
        )
    )

    # ==========================================================
    # Generate project-day spine
    # ==========================================================

    project_days = []

    for _, project in projects.iterrows():

        start = project[
            "planned_start_date"
        ]

        end = project[
            "planned_end_date"
        ]

        if pd.isna(start) or pd.isna(end):
            continue

        dates = pd.date_range(
            start=start,
            end=end,
            freq="D",
        )

        for date in dates:

            project_days.append(
                {
                    "project_id": project[
                        "project_id"
                    ],
                    "date": date,
                }
            )

    project_daily = pd.DataFrame(
        project_days
    )

    # ==========================================================
    # Merge operational signals
    # ==========================================================

    project_daily = project_daily.merge(
        activity_daily,
        on=[
            "project_id",
            "date",
        ],
        how="left",
    )

    project_daily = project_daily.merge(
        material_daily,
        on=[
            "project_id",
            "date",
        ],
        how="left",
    )

    project_daily = project_daily.merge(
        delivery_daily,
        on=[
            "project_id",
            "date",
        ],
        how="left",
    )

    project_daily = project_daily.merge(
        equipment_daily,
        on="date",
        how="left",
    )

    # ==========================================================
    # Fill missing operational values
    # ==========================================================

    numeric_columns = [
        "active_activity_count",
        "critical_activity_count",
        "delayed_activity_count",
        "activity_delay_days",
        "baseline_activity_cost",
        "actual_activity_cost",
        "planned_material_quantity",
        "actual_material_quantity",
        "material_stockout_count",
        "material_records",
        "delivery_count",
        "delivered_quantity",
        "equipment_operating_hours",
        "equipment_downtime_hours",
        "equipment_failure_count",
        "equipment_maintenance_count",
        "equipment_maintenance_cost",
    ]

    for column in numeric_columns:

        if column in project_daily.columns:

            project_daily[column] = (
                project_daily[column]
                .fillna(0)
            )

    # ==========================================================
    # Project-level fields
    # ==========================================================

    project_fields = projects[
        [
            "project_id",
            "contract_value",
            "planned_start_date",
            "planned_end_date",
            "actual_start_date",
            "planned_duration_days",
            "project_complexity",
            "delay_days",
            "delay_flag",
        ]
    ].copy()

    project_daily = project_daily.merge(
        project_fields,
        on="project_id",
        how="left",
    )

    # ==========================================================
    # Schedule progress signals
    # ==========================================================

    project_daily["elapsed_days"] = (
        project_daily["date"]
        - project_daily["planned_start_date"]
    ).dt.days

    project_daily["elapsed_days"] = (
        project_daily["elapsed_days"]
        .clip(lower=0)
    )

    project_daily["planned_progress_pct"] = (
        project_daily["elapsed_days"]
        / project_daily[
            "planned_duration_days"
        ].clip(lower=1)
        * 100
    )

    project_daily[
        "planned_progress_pct"
    ] = (
        project_daily[
            "planned_progress_pct"
        ].clip(
            lower=0,
            upper=100,
        )
    )

    # ==========================================================
    # Actual progress proxy
    #
    # Derived from activity completion signals.
    # Does NOT use final project delay.
    # ==========================================================

    project_daily[
        "daily_activity_intensity"
    ] = (
        project_daily[
            "active_activity_count"
        ]
        + 0.5
        * project_daily[
            "critical_activity_count"
        ]
    )

    project_daily[
        "progress_productivity_index"
    ] = (
        project_daily[
            "daily_activity_intensity"
        ]
        / project_daily[
            "active_activity_count"
        ].replace(
            0,
            np.nan,
        )
    ).fillna(0)

    # ==========================================================
    # Material availability signal
    # ==========================================================

    project_daily[
        "material_availability_ratio"
    ] = (
        project_daily[
            "delivered_quantity"
        ]
        / project_daily[
            "planned_material_quantity"
        ].replace(
            0,
            np.nan,
        )
    ).fillna(0)

    project_daily[
        "material_availability_ratio"
    ] = (
        project_daily[
            "material_availability_ratio"
        ].clip(
            lower=0,
            upper=5,
        )
    )

    # ==========================================================
    # Cost signals
    # ==========================================================

    project_daily[
        "cost_variance"
    ] = (
        project_daily[
            "actual_activity_cost"
        ]
        - project_daily[
            "baseline_activity_cost"
        ]
    )

    project_daily[
        "cost_variance_pct"
    ] = (
        project_daily[
            "cost_variance"
        ]
        / project_daily[
            "baseline_activity_cost"
        ].replace(
            0,
            np.nan,
        )
        * 100
    ).fillna(0)

    # ==========================================================
    # Operational risk signals
    # ==========================================================

    project_daily[
        "operational_risk_score"
    ] = (
        0.30
        * np.clip(
            project_daily[
                "delayed_activity_count"
            ]
            / project_daily[
                "active_activity_count"
            ].replace(
                0,
                np.nan,
            ),
            0,
            1,
        ).fillna(0)
        +
        0.30
        * np.clip(
            project_daily[
                "material_stockout_count"
            ]
            / project_daily[
                "material_records"
            ].replace(
                0,
                np.nan,
            ),
            0,
            1,
        ).fillna(0)
        +
        0.20
        * np.clip(
            project_daily[
                "equipment_downtime_hours"
            ]
            / 24,
            0,
            1,
        )
        +
        0.20
        * np.clip(
            project_daily[
                "cost_variance_pct"
            ]
            / 100,
            0,
            1,
        )
    )

    # ==========================================================
    # Future ML target
    #
    # IMPORTANT:
    # delay_flag is NOT a feature.
    #
    # It is preserved as the eventual target.
    # ==========================================================

    project_daily[
        "project_delay_target"
    ] = project_daily[
        "delay_flag"
    ].astype(int)

    # ==========================================================
    # Clean up
    # ==========================================================

    project_daily = project_daily.sort_values(
        [
            "project_id",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    # ==========================================================
    # Daily operational deterioration signals
    # ==========================================================

    project_daily[
        "delay_signal_flag"
    ] = (
        (
            project_daily[
                "delayed_activity_count"
            ] > 0
        )
        |
        (
            project_daily[
                "material_stockout_count"
            ] > 0
        )
        |
        (
            project_daily[
                "cost_variance_pct"
            ] > 10
        )
    ).astype(int)

    # ==========================================================
    # Generate record IDs
    # ==========================================================

    project_daily.insert(
        0,
        "performance_id",
        [
            f"PERF-{i:08d}"
            for i in range(
                1,
                len(project_daily) + 1,
            )
        ],
    )

    return project_daily