import pandas as pd


LAG_COLUMNS = [
    "active_activity_count",
    "critical_activity_count",
    "delayed_activity_count",
    "activity_delay_days",
    "material_stockout_count",
    "delivery_count",
    "equipment_downtime_hours",
    "equipment_failure_count",
    "equipment_maintenance_cost",
    "cost_variance",
    "cost_variance_pct",
]


ROLLING_COLUMNS = [
    "activity_delay_days",
    "material_stockout_count",
    "delivery_count",
    "equipment_downtime_hours",
    "equipment_failure_count",
    "cost_variance",
    "cost_variance_pct",
]


def prepare_temporal_data(
    df: pd.DataFrame,
    group_column: str = "project_id",
    date_column: str = "date",
) -> pd.DataFrame:
    """Prepare project daily data for temporal feature engineering."""

    result = df.copy()

    result[date_column] = pd.to_datetime(
        result[date_column]
    )

    result = result.sort_values(
        [group_column, date_column]
    ).reset_index(drop=True)

    return result


def add_lag_features(
    df: pd.DataFrame,
    group_column: str = "project_id",
) -> pd.DataFrame:
    """Add historical lag features without future leakage."""

    result = df.copy()

    for column in LAG_COLUMNS:

        if column not in result.columns:
            continue

        grouped = result.groupby(
            group_column,
            sort=False,
        )[column]

        result[f"{column}_lag_1"] = grouped.shift(1)
        result[f"{column}_lag_7"] = grouped.shift(7)

    return result


def add_rolling_features(
    df: pd.DataFrame,
    group_column: str = "project_id",
) -> pd.DataFrame:
    """
    Add historical rolling features.

    Current-day values are excluded using shift(1).
    """

    result = df.copy()

    for column in ROLLING_COLUMNS:

        if column not in result.columns:
            continue

        historical = (
            result.groupby(
                group_column,
                sort=False,
            )[column]
            .shift(1)
        )

        for window in [7, 14, 30]:

            result[
                f"{column}_rolling_{window}d"
            ] = (
                historical
                .groupby(
                    result[group_column],
                    sort=False,
                )
                .transform(
                    lambda x: x.rolling(
                        window=window,
                        min_periods=1,
                    ).mean()
                )
            )

    return result


def build_temporal_features(
    df: pd.DataFrame,
    group_column: str = "project_id",
    date_column: str = "date",
) -> pd.DataFrame:
    """Build the complete temporal feature layer."""

    result = prepare_temporal_data(
        df,
        group_column=group_column,
        date_column=date_column,
    )

    result = add_lag_features(
        result,
        group_column=group_column,
    )

    result = add_rolling_features(
        result,
        group_column=group_column,
    )

    return result