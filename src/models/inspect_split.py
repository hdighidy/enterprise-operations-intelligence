from src.models.dataset import load_ml_dataset
from src.models.split import time_based_split


def main():

    df = load_ml_dataset()

    train, validation, test = time_based_split(
        df
    )

    print("=" * 70)
    print("TIME-BASED ML SPLIT")
    print("=" * 70)

    print("\nTRAIN")
    print(f"Records : {len(train):,}")
    print(
        f"Dates   : "
        f"{train['date'].min()} → "
        f"{train['date'].max()}"
    )
    print(
        f"Target  : "
        f"{train['delay_next_30_days'].mean() * 100:.2f}%"
    )

    print("\nVALIDATION")
    print(f"Records : {len(validation):,}")
    print(
        f"Dates   : "
        f"{validation['date'].min()} → "
        f"{validation['date'].max()}"
    )
    print(
        f"Target  : "
        f"{validation['delay_next_30_days'].mean() * 100:.2f}%"
    )

    print("\nTEST")
    print(f"Records : {len(test):,}")
    print(
        f"Dates   : "
        f"{test['date'].min()} → "
        f"{test['date'].max()}"
    )
    print(
        f"Target  : "
        f"{test['delay_next_30_days'].mean() * 100:.2f}%"
    )

    print("\nPROJECT DISTRIBUTION")

    print(
        "Train projects      : "
        f"{train['project_id'].nunique():,}"
    )

    print(
        "Validation projects : "
        f"{validation['project_id'].nunique():,}"
    )

    print(
        "Test projects       : "
        f"{test['project_id'].nunique():,}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()