from __future__ import annotations

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a binary classifier using predicted probabilities.

    The threshold is configurable so that threshold tuning can later
    be performed on the validation set without changing the model.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "threshold": threshold,
    }