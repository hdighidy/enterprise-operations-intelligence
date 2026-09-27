from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline


def extract_logistic_coefficients(
    model: Pipeline,
) -> pd.DataFrame:
    """
    Extract Logistic Regression coefficients from a fitted pipeline.

    Returns a DataFrame containing:
        feature
        coefficient
        absolute_coefficient
        direction
    """

    if not isinstance(model, Pipeline):
        raise TypeError("model must be a sklearn Pipeline.")

    if "preprocessor" not in model.named_steps:
        raise ValueError("Pipeline must contain a 'preprocessor' step.")

    if "classifier" not in model.named_steps:
        raise ValueError("Pipeline must contain a 'classifier' step.")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        raise ValueError(
            "The classifier must expose Logistic Regression coefficients."
        )

    feature_names = preprocessor.get_feature_names_out()

    coefficients = classifier.coef_[0]

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Number of feature names does not match "
            "number of model coefficients."
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    result["absolute_coefficient"] = (
        result["coefficient"].abs()
    )

    result["direction"] = result["coefficient"].apply(
        lambda value: (
            "positive"
            if value > 0
            else "negative"
            if value < 0
            else "neutral"
        )
    )

    result = result.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    return result


def save_feature_importance(
    coefficients: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save extracted model coefficients to CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coefficients.to_csv(
        output_path,
        index=False,
    )