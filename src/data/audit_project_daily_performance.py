from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import RAW_DATA_DIR


INPUT_FILE = RAW_DATA_DIR / "project_daily_performance.csv"


REQUIRED_COLUMNS = [
    "performance_id",
    "project_id",
    "date",
    "active_activity_count",
    "critical_activity_count",
    "delayed_activity_count",
    "activity_delay_days",
    "baseline_activity_cost",
    "actual_activity_cost",
    "planned_material_quantity",
    "actual_material_quantity",
    "material_stockout_count",
    "delivery_count",
    "delivered_quantity",
    "equipment_operating_hours",
    "equipment_downtime_hours",
    "equipment_maintenance_count",
    "equipment_failure_count",
    "equipment_maintenance_cost",
    "planned_progress_pct",
    "daily_activity_intensity",
    "progress_productivity_index",
    "material_availability_ratio",
    "cost_variance",
    "cost_variance_pct",
    "operational_risk_score",
    "delay_signal_flag",
    "project_delay_target",
]


def check_required_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]


def audit_data(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("PROJECT DAILY PERFORMANCE — STEP 8.5 AUDIT")
    print("=" * 70)

    print(f"Records              : {len(df):,}")
    print(f"Projects             : {df['project_id'].nunique():,}")
    print(f"Date range           : {df['date'].min()} → {df['date'].max()}")
    print(f"Columns              : {len(df.columns)}")

    print("\n--- Required Columns ---")

    missing_columns = check_required_columns(df)

    if missing_columns:
        print("MISSING:")
        for column in missing_columns:
            print(f"  - {column}")
    else:
        print("All required columns present.")

    print("\n--- Duplicate Checks ---")

    print(
        "Duplicate performance_id:",
        df["performance_id"].duplicated().sum(),
    )

    print(
        "Duplicate project/date:",
        df.duplicated(["project_id", "date"]).sum(),
    )

    print("\n--- Missing Values ---")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(missing.to_string())

    print("\n--- Numeric Range Checks ---")

    range_checks = {
        "planned_progress_pct": (0, 100),
        "material_availability_ratio": (0, None),
        "operational_risk_score": (0, 1),
    }

    for column, (minimum, maximum) in range_checks.items():

        if column not in df.columns:
            continue

        series = df[column]

        below_min = (
            series < minimum
        ).sum()

        above_max = (
            series > maximum
        ).sum() if maximum is not None else 0

        print(
            f"{column:35s}"
            f" below_min={below_min:,}"
            f" above_max={above_max:,}"
        )

    print("\n--- Negative Operational Values ---")

    non_negative_columns = [
        "active_activity_count",
        "critical_activity_count",
        "delayed_activity_count",
        "activity_delay_days",
        "baseline_activity_cost",
        "actual_activity_cost",
        "planned_material_quantity",
        "actual_material_quantity",
        "material_stockout_count",
        "delivery_count",
        "delivered_quantity",
        "equipment_operating_hours",
        "equipment_downtime_hours",
        "equipment_maintenance_count",
        "equipment_failure_count",
        "equipment_maintenance_cost",
    ]

    for column in non_negative_columns:

        if column not in df.columns:
            continue

        negative_count = (df[column] < 0).sum()

        if negative_count:
            print(
                f"{column:35s}: {negative_count:,} negative values"
            )

    print("\n--- Binary Flag Checks ---")

    binary_columns = [
        "delay_signal_flag",
        "project_delay_target",
    ]

    for column in binary_columns:

        if column not in df.columns:
            continue

        values = sorted(
            df[column]
            .dropna()
            .unique()
            .tolist()
        )

        print(f"{column:30s}: {values}")

    print("\n--- Distribution ---")

    if "project_delay_target" in df.columns:
        print(
            "Final project delay target rate:",
            f"{df['project_delay_target'].mean() * 100:.2f}%"
        )

    if "delay_signal_flag" in df.columns:
        print(
            "Operational delay signal rate:",
            f"{df['delay_signal_flag'].mean() * 100:.2f}%"
        )

    print("\n--- Cost Statistics ---")

    cost_columns = [
        "baseline_activity_cost",
        "actual_activity_cost",
        "cost_variance",
        "cost_variance_pct",
    ]

    available_cost_columns = [
        column
        for column in cost_columns
        if column in df.columns
    ]

    if available_cost_columns:
        print(
            df[available_cost_columns]
            .describe()
            .round(2)
            .to_string()
        )

    print("\n--- Project Date Continuity ---")

    df_sorted = df.sort_values(
        ["project_id", "date"]
    )

    gaps = []

    for project_id, group in df_sorted.groupby("project_id"):

        dates = pd.to_datetime(group["date"]).sort_values()

        if len(dates) < 2:
            continue

        differences = dates.diff().dropna().dt.days

        gap_count = (differences > 1).sum()

        if gap_count:
            gaps.append(
                {
                    "project_id": project_id,
                    "gap_count": int(gap_count),
                }
            )

    if gaps:
        print(
            f"Projects with date gaps: {len(gaps)}"
        )
        print(
            pd.DataFrame(gaps)
            .head(10)
            .to_string(index=False)
        )
    else:
        print("No date gaps detected.")

    print("\n--- Daily Activity Signal ---")

    if "active_activity_count" in df.columns:

        print(
            "Days with active activities:",
            (df["active_activity_count"] > 0).mean() * 100,
            "%",
        )

    if "critical_activity_count" in df.columns:

        print(
            "Days with critical activities:",
            (df["critical_activity_count"] > 0).mean() * 100,
            "%",
        )

    print("\n--- Equipment Signal ---")

    if "equipment_failure_count" in df.columns:

        print(
            "Days with equipment failures:",
            (df["equipment_failure_count"] > 0).mean() * 100,
            "%",
        )

    if "equipment_downtime_hours" in df.columns:

        print(
            "Total equipment downtime:",
            f"{df['equipment_downtime_hours'].sum():,.2f} hours",
        )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    audit_data(df)


if __name__ == "__main__":
    main()