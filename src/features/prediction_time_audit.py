from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class AuditResult:
    category: str
    feature: str
    status: str
    severity: str
    explanation: str


# -------------------------------------------------------------------
# Explicitly forbidden features
# -------------------------------------------------------------------

FORBIDDEN_FEATURES = {
    "project_delay_target",
    "actual_finish_date",
    "actual_duration_days",
    "final_delay_days",
    "delay_next_30_days",
}


# -------------------------------------------------------------------
# Features that require semantic review
# -------------------------------------------------------------------

REVIEW_FEATURES = {
    "activity_delay_days": (
        "Current observed activity delay. "
        "Potentially valid at prediction time, but highly correlated "
        "with the future target in the synthetic dataset."
    ),
    "delay_signal_flag": (
        "Derived operational delay signal. "
        "Requires verification that its calculation does not use "
        "future or final project outcomes."
    ),
    "operational_risk_score": (
        "Composite risk indicator. "
        "Requires verification of all underlying components and "
        "their prediction-time availability."
    ),
}


# -------------------------------------------------------------------
# Columns that are expected to be historical / lagged
# -------------------------------------------------------------------

HISTORICAL_FEATURE_PATTERNS = (
    "_lag_1",
    "_lag_7",
    "_rolling_7d",
    "_rolling_14d",
    "_rolling_30d",
)


def audit_feature_names(
    df: pd.DataFrame,
) -> List[AuditResult]:
    """
    Audit feature names for obvious target leakage.
    """

    results: List[AuditResult] = []

    for feature in df.columns:

        if feature in FORBIDDEN_FEATURES:
            results.append(
                AuditResult(
                    category="Leakage",
                    feature=feature,
                    status="FAIL",
                    severity="HIGH",
                    explanation=(
                        "Feature is explicitly forbidden because "
                        "it contains final outcome or future-target information."
                    ),
                )
            )

        elif feature in REVIEW_FEATURES:
            results.append(
                AuditResult(
                    category="Semantic Review",
                    feature=feature,
                    status="REVIEW",
                    severity="MEDIUM",
                    explanation=REVIEW_FEATURES[feature],
                )
            )

        elif feature.endswith(HISTORICAL_FEATURE_PATTERNS):
            results.append(
                AuditResult(
                    category="Temporal",
                    feature=feature,
                    status="PASS",
                    severity="LOW",
                    explanation=(
                        "Feature is explicitly represented as a "
                        "historical lag or rolling-window feature."
                    ),
                )
            )

    return results


def audit_target_construction(
    df: pd.DataFrame,
    target_column: str = "delay_next_30_days",
) -> List[AuditResult]:
    """
    Validate basic properties of the future-delay target.
    """

    results: List[AuditResult] = []

    if target_column not in df.columns:
        results.append(
            AuditResult(
                category="Target",
                feature=target_column,
                status="FAIL",
                severity="HIGH",
                explanation="Target column is missing.",
            )
        )
        return results

    unique_values = set(
        pd.Series(df[target_column]).dropna().unique()
    )

    if unique_values.issubset({0, 1}):
        results.append(
            AuditResult(
                category="Target",
                feature=target_column,
                status="PASS",
                severity="LOW",
                explanation="Target is binary.",
            )
        )
    else:
        results.append(
            AuditResult(
                category="Target",
                feature=target_column,
                status="FAIL",
                severity="HIGH",
                explanation=(
                    f"Target contains unexpected values: "
                    f"{sorted(unique_values)}"
                ),
            )
        )

    return results


def audit_missing_values(
    df: pd.DataFrame,
) -> List[AuditResult]:
    """
    Report missing values in model features.

    Missing historical values are expected at the beginning
    of project histories because lag/rolling features require
    prior observations.
    """

    results: List[AuditResult] = []

    missing_counts = df.isna().sum()

    for feature, count in missing_counts[missing_counts > 0].items():

        results.append(
            AuditResult(
                category="Missingness",
                feature=feature,
                status="REVIEW",
                severity="LOW",
                explanation=(
                    f"{int(count):,} missing values detected. "
                    "These should be handled inside the model preprocessing pipeline."
                ),
            )
        )

    if not results:
        results.append(
            AuditResult(
                category="Missingness",
                feature="dataset",
                status="PASS",
                severity="LOW",
                explanation="No missing values detected.",
            )
        )

    return results


def run_prediction_time_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Execute the complete prediction-time feature audit.
    """

    results = []

    results.extend(audit_feature_names(df))
    results.extend(audit_target_construction(df))
    results.extend(audit_missing_values(df))

    return pd.DataFrame(
        [
            {
                "category": result.category,
                "feature": result.feature,
                "status": result.status,
                "severity": result.severity,
                "explanation": result.explanation,
            }
            for result in results
        ]
    )


def print_audit_report(
    audit_df: pd.DataFrame,
) -> None:

    print("=" * 70)
    print("STEP 10.6 — PREDICTION-TIME FEATURE AUDIT")
    print("=" * 70)

    print("\nAudit summary")
    print("-" * 70)

    summary = (
        audit_df.groupby(["category", "status"])
        .size()
        .reset_index(name="count")
    )

    print(summary.to_string(index=False))

    print("\nHigh-severity findings")
    print("-" * 70)

    high = audit_df[
        audit_df["severity"] == "HIGH"
    ]

    if high.empty:
        print("None")
    else:
        print(
            high[
                [
                    "category",
                    "feature",
                    "status",
                    "explanation",
                ]
            ].to_string(index=False)
        )

    print("\nSemantic review items")
    print("-" * 70)

    review = audit_df[
        audit_df["status"] == "REVIEW"
    ]

    if review.empty:
        print("None")
    else:
        print(
            review[
                [
                    "category",
                    "feature",
                    "explanation",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":

    input_path = "data/processed/ml_dataset.csv"

    df = pd.read_csv(input_path)

    audit_df = run_prediction_time_audit(df)

    print_audit_report(audit_df)