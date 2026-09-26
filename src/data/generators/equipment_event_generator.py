from __future__ import annotations

import numpy as np
import pandas as pd


def generate_equipment_events(
    equipment: pd.DataFrame,
    seed: int = 42,
    max_equipment: int = 500,
    days: int = 180,
) -> pd.DataFrame:
    """
    Generate daily equipment operational events.

    Development configuration:
        500 equipment
        180 days
        ~90,000 records

    The generated data supports future:
        - predictive maintenance
        - failure prediction
        - equipment utilization analysis
        - downtime forecasting
        - maintenance cost analysis
    """

    rng = np.random.default_rng(seed)

    # ==========================================================
    # Validate schema
    # ==========================================================

    required_columns = {
        "equipment_id",
        "equipment_type",
        "operating_hours",
        "downtime_hours",
        "utilization_rate",
        "maintenance_cost",
    }

    missing = (
        required_columns
        - set(equipment.columns)
    )

    if missing:
        raise ValueError(
            "Missing equipment columns: "
            f"{sorted(missing)}"
        )

    equipment = equipment.copy()

    # ==========================================================
    # Limit equipment for local development
    # ==========================================================

    if len(equipment) > max_equipment:

        equipment = equipment.sample(
            n=max_equipment,
            random_state=seed,
        )

    equipment = equipment.reset_index(
        drop=True
    )

    # ==========================================================
    # Equipment-type characteristics
    # ==========================================================

    type_config = {
        "Crane": {
            "failure_factor": 1.20,
            "utilization": 0.72,
            "maintenance_factor": 1.30,
        },
        "Excavator": {
            "failure_factor": 1.15,
            "utilization": 0.68,
            "maintenance_factor": 1.20,
        },
        "Concrete Pump": {
            "failure_factor": 1.10,
            "utilization": 0.65,
            "maintenance_factor": 1.15,
        },
        "Welding Machine": {
            "failure_factor": 0.85,
            "utilization": 0.58,
            "maintenance_factor": 0.90,
        },
        "Compressor": {
            "failure_factor": 0.95,
            "utilization": 0.62,
            "maintenance_factor": 0.95,
        },
    }

    default_config = {
        "failure_factor": 1.0,
        "utilization": 0.60,
        "maintenance_factor": 1.0,
    }

    records = []

    # Fixed development period
    start_date = pd.Timestamp(
        "2025-01-01"
    )

    event_counter = 1

    # ==========================================================
    # Generate daily history
    # ==========================================================

    for _, row in equipment.iterrows():

        equipment_id = row[
            "equipment_id"
        ]

        equipment_type = row[
            "equipment_type"
        ]

        config = type_config.get(
            equipment_type,
            default_config,
        )

        failure_factor = config[
            "failure_factor"
        ]

        base_utilization = config[
            "utilization"
        ]

        maintenance_factor = config[
            "maintenance_factor"
        ]

        # ------------------------------------------------------
        # Historical operating condition
        # ------------------------------------------------------

        historical_hours = float(
            row["operating_hours"]
        )

        historical_downtime = float(
            row["downtime_hours"]
        )

        historical_maintenance = float(
            row["maintenance_cost"]
        )

        # ------------------------------------------------------
        # Equipment-specific latent risk
        # ------------------------------------------------------

        age_risk = min(
            historical_hours / 20000.0,
            1.0,
        )

        downtime_risk = min(
            historical_downtime / 100.0,
            1.0,
        )

        equipment_risk = (
            0.45 * age_risk
            + 0.35 * downtime_risk
            + 0.20 * (
                1.0
                - float(
                    row["utilization_rate"]
                )
            )
        )

        equipment_risk = np.clip(
            equipment_risk,
            0.01,
            0.95,
        )

        # ------------------------------------------------------
        # Daily history
        # ------------------------------------------------------

        for day in range(days):

            event_date = (
                start_date
                + pd.Timedelta(days=day)
            )

            # --------------------------------------------------
            # Daily utilization
            # --------------------------------------------------

            utilization = (
                base_utilization
                + rng.normal(
                    0,
                    0.08,
                )
            )

            utilization = np.clip(
                utilization,
                0.10,
                0.98,
            )

            # Weekend reduction
            if event_date.weekday() >= 5:

                utilization *= 0.85

            utilization = round(
                utilization,
                4,
            )

            # --------------------------------------------------
            # Operating hours
            # --------------------------------------------------

            operating_hours = (
                utilization * 10.0
            )

            operating_hours *= (
                rng.uniform(
                    0.90,
                    1.10,
                )
            )

            operating_hours = round(
                max(
                    0,
                    operating_hours,
                ),
                2,
            )

            # --------------------------------------------------
            # Downtime
            # --------------------------------------------------

            downtime_hours = (
                max(
                    0,
                    rng.gamma(
                        shape=1.5,
                        scale=1.2,
                    )
                )
            )

            downtime_hours *= (
                1
                + equipment_risk
            )

            downtime_hours = round(
                downtime_hours,
                2,
            )

            # --------------------------------------------------
            # Maintenance event
            # --------------------------------------------------

            maintenance_probability = (
                0.025
                + 0.08 * equipment_risk
                + 0.02 * failure_factor
            )

            maintenance_flag = int(
                rng.random()
                < maintenance_probability
            )

            # --------------------------------------------------
            # Failure event
            # --------------------------------------------------

            failure_probability = (
                0.008
                + 0.06 * equipment_risk
                + 0.025 * failure_factor
            )

            # Higher utilization increases failure risk
            failure_probability += (
                0.03
                * max(
                    utilization - 0.70,
                    0,
                )
            )

            failure_probability = np.clip(
                failure_probability,
                0.001,
                0.30,
            )

            failure_flag = int(
                rng.random()
                < failure_probability
            )

            # --------------------------------------------------
            # Failure severity
            # --------------------------------------------------

            if failure_flag:

                failure_severity = rng.choice(
                    [
                        "Minor",
                        "Moderate",
                        "Major",
                        "Critical",
                    ],
                    p=[
                        0.45,
                        0.30,
                        0.20,
                        0.05,
                    ],
                )

            else:

                failure_severity = "None"

            # --------------------------------------------------
            # Maintenance cost
            # --------------------------------------------------

            daily_maintenance_cost = (
                historical_maintenance
                / max(
                    days,
                    1,
                )
            )

            daily_maintenance_cost *= (
                maintenance_factor
            )

            daily_maintenance_cost *= (
                rng.uniform(
                    0.5,
                    1.5,
                )
            )

            if maintenance_flag:

                daily_maintenance_cost *= (
                    rng.uniform(
                        2.0,
                        5.0,
                    )
                )

            if failure_flag:

                daily_maintenance_cost *= (
                    rng.uniform(
                        3.0,
                        10.0,
                    )
                )

            daily_maintenance_cost = round(
                max(
                    0,
                    daily_maintenance_cost,
                ),
                2,
            )

            # --------------------------------------------------
            # Event type
            # --------------------------------------------------

            if failure_flag:

                event_type = "Failure"

            elif maintenance_flag:

                event_type = "Maintenance"

            elif downtime_hours > 4:

                event_type = "Downtime"

            else:

                event_type = "Normal Operation"

            # --------------------------------------------------
            # Remaining useful-life proxy
            # --------------------------------------------------

            degradation_index = (
                0.40 * age_risk
                + 0.30 * downtime_risk
                + 0.30 * utilization
            )

            degradation_index = np.clip(
                degradation_index,
                0,
                1,
            )

            # --------------------------------------------------
            # Record
            # --------------------------------------------------

            records.append(
                {
                    "event_id": (
                        f"EQEV-{event_counter:08d}"
                    ),
                    "event_date": (
                        event_date.date()
                    ),
                    "equipment_id": (
                        equipment_id
                    ),
                    "equipment_type": (
                        equipment_type
                    ),
                    "operating_hours": (
                        operating_hours
                    ),
                    "downtime_hours": (
                        downtime_hours
                    ),
                    "utilization_rate": (
                        utilization
                    ),
                    "maintenance_flag": (
                        maintenance_flag
                    ),
                    "failure_flag": (
                        failure_flag
                    ),
                    "failure_severity": (
                        failure_severity
                    ),
                    "maintenance_cost": (
                        daily_maintenance_cost
                    ),
                    "degradation_index": round(
                        degradation_index,
                        4,
                    ),
                    "event_type": (
                        event_type
                    ),
                }
            )

            event_counter += 1

    return pd.DataFrame(records)