import pandas as pd


def create_future_delay_target(
    df: pd.DataFrame,
    group_column: str = "project_id",
    date_column: str = "date",
    delay_column: str = "activity_delay_days",
    horizon_days: int = 30,
    delay_threshold_days: int = 3,
) -> pd.DataFrame:
    """
    Create a future delay target.

    For each project-day T:

        delay_next_30_days = 1

    if a qualifying delay occurs during the following
    horizon_days.

    The current day's delay is NOT included.
    """

    if horizon_days < 1:
        raise ValueError(
            "horizon_days must be >= 1"
        )

    if delay_threshold_days < 1:
        raise ValueError(
            "delay_threshold_days must be >= 1"
        )

    required_columns = {
        group_column,
        date_column,
        delay_column,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    result[date_column] = pd.to_datetime(
        result[date_column]
    )

    result = result.sort_values(
        [group_column, date_column]
    ).reset_index(drop=True)

    result["delay_next_30_days"] = 0

    for _, group in result.groupby(
        group_column,
        sort=False,
    ):
        indices = group.index

        # Convert delay into a binary future-delay event.
        delay_event = (
            group[delay_column]
            .gt(delay_threshold_days)
            .astype(int)
        )

        # Exclude today's event.
        future_events = delay_event.shift(-1)

        # Reverse the series so rolling() looks forward
        # in the original chronological direction.
        future_target = (
            future_events
            .iloc[::-1]
            .rolling(
                window=horizon_days,
                min_periods=1,
            )
            .max()
            .iloc[::-1]
        )

        result.loc[
            indices,
            "delay_next_30_days",
        ] = (
            future_target
            .fillna(0)
            .astype(int)
            .values
        )

    return result