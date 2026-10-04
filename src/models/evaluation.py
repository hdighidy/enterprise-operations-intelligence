from __future__ import annotations

import numpy as np
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
    Evaluate a binary classification model.

    The model must expose predict_proba().
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    probabilities = np.asarray(
        model.predict_proba(X)
    )

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "predict_proba() must return a 2-dimensional "
            "array with at least two probability columns."
        )

    probabilities = probabilities[:, 1]

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
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": threshold,
    }