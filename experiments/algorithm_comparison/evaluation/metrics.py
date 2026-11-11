"""Metrics that match the existing training conventions."""
from __future__ import annotations

from typing import Any

import numpy as np


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def binary_forecast_metrics(y_true: Any, y_pred: Any, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    truth = np.asarray(y_true).astype(bool).ravel()
    pred = (np.asarray(y_pred).ravel() >= threshold)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[False, True]).ravel()
    denominator = tp + fn
    far_denominator = tp + fp
    union = tp + fp + fn
    return {
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "csi": float(tp / (union or 1)),
        "pod": float(tp / (denominator or 1)),
        "far": float(fp / (far_denominator or 1)),
        "hss": float(2 * (tp * tn - fn * fp) / (((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn)) or 1)),
        "iou": float(tp / (union or 1)),
    }
