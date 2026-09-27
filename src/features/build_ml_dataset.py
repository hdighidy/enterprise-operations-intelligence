from pathlib import Path

import pandas as pd

from src.features.temporal_features import build_temporal_features
from src.features.target_engineering import create_future_delay_target


RAW_DATA_PATH = Path(
    "data/raw/project_daily_performance.csv"
)

OUTPUT_PATH = Path(
    "data/processed/ml_dataset.csv"
)


LEAKAGE_COLUMNS = {
    "project_delay_target",
    "actual_finish_date",
    "actual_duration_days",
    "final_delay_days",
}


def build_ml_dataset(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(
            "Input dataset is empty."
        )

    # 1. Historical temporal features
    result = build_temporal_features(
        df,
        group_column="project_id",
        date_column="date",
    )

    # 2. Future target
    result = create_future_delay_target(
        result,
        group_column="project_id",
        date_column="date",
        delay_column="activity_delay_days",
        horizon_days=30,
        delay_threshold_days=3,
    )

    # 3. Remove leakage columns
    leakage_to_remove = [
        column
        for column in LEAKAGE_COLUMNS
        if column in result.columns
    ]

    result = result.drop(
        columns=leakage_to_remove
    )

    # 4. Keep target explicitly
    if "delay_next_30_days" not in result.columns:
        raise AssertionError(
            "Future target was not generated."
        )

    # 5. Sort
    result = result.sort_values(
        ["project_id", "date"]
    ).reset_index(drop=True)

    # 6. Save
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return result


if __name__ == "__main__":

    result = build_ml_dataset()

    print("=" * 70)
    print("LEAKAGE-CONTROLLED ML DATASET")
    print("=" * 70)

    print(f"Records       : {len(result):,}")

    print(
        f"Projects      : "
        f"{result['project_id'].nunique():,}"
    )

    print(
        f"Features      : "
        f"{len(result.columns):,}"
    )

    print(
        f"Target rate   : "
        f"{result['delay_next_30_days'].mean() * 100:.2f}%"
    )

    print(
        f"Output        : {OUTPUT_PATH}"
    )

    print("=" * 70)