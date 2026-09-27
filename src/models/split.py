import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    date_column: str = "date",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
):
    """
    Chronological train/validation/test split.

    The split is performed using unique dates rather than
    individual rows.

    This prevents records from the same calendar date from
    being distributed across different datasets.
    """

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be < 1."
        )

    if date_column not in df.columns:
        raise ValueError(
            f"Missing date column: {date_column}"
        )

    result = df.copy()

    result[date_column] = pd.to_datetime(
        result[date_column]
    )

    unique_dates = sorted(
        result[date_column].dropna().unique()
    )

    if len(unique_dates) < 3:
        raise ValueError(
            "At least 3 unique dates are required."
        )

    n_dates = len(unique_dates)

    train_end = int(
        n_dates * train_ratio
    )

    validation_end = int(
        n_dates * (
            train_ratio + validation_ratio
        )
    )

    train_dates = unique_dates[
        :train_end
    ]

    validation_dates = unique_dates[
        train_end:validation_end
    ]

    test_dates = unique_dates[
        validation_end:
    ]

    train = result[
        result[date_column].isin(train_dates)
    ].copy()

    validation = result[
        result[date_column].isin(
            validation_dates
        )
    ].copy()

    test = result[
        result[date_column].isin(test_dates)
    ].copy()

    return (
        train.sort_values(
            date_column
        ).reset_index(drop=True),

        validation.sort_values(
            date_column
        ).reset_index(drop=True),

        test.sort_values(
            date_column
        ).reset_index(drop=True),
    )