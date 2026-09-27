from pathlib import Path

from src.models.baseline import build_logistic_pipeline
from src.models.dataset import (
    load_ml_dataset,
    prepare_features_and_target,
)
from src.models.explainability import (
    extract_logistic_coefficients,
    save_feature_importance,
)
from src.models.preprocessing import get_model_columns
from src.models.split import time_based_split


INPUT_PATH = Path(
    "data/processed/ml_dataset.csv"
)

OUTPUT_PATH = Path(
    "data/processed/logistic_feature_importance.csv"
)


def main():
    df = load_ml_dataset(INPUT_PATH)

    train_df, _, _ = time_based_split(
        df,
        date_column="date",
    )

    X_train, y_train = prepare_features_and_target(
        train_df
    )

    numeric_columns, categorical_columns = (
        get_model_columns(X_train)
    )

    model = build_logistic_pipeline(
        numeric_columns,
        categorical_columns,
    )

    model.fit(
        X_train,
        y_train,
    )

    coefficients = extract_logistic_coefficients(
        model
    )

    save_feature_importance(
        coefficients,
        OUTPUT_PATH,
    )

    print("=" * 70)
    print("LOGISTIC REGRESSION — FEATURE COEFFICIENTS")
    print("=" * 70)

    print(
        f"Total transformed features : "
        f"{len(coefficients):,}"
    )

    print("\nTOP 15 POSITIVE FEATURES")
    print("-" * 70)

    print(
        coefficients[
            coefficients["coefficient"] > 0
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nTOP 15 NEGATIVE FEATURES")
    print("-" * 70)

    print(
        coefficients[
            coefficients["coefficient"] < 0
        ]
        .sort_values(
            "coefficient",
            ascending=True,
        )
        .head(15)
        .to_string(index=False)
    )

    print("\nOutput:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()