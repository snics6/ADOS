"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.count_nonzero(m) < 3:
        return float("nan")
    if np.unique(y_true[m]).size < 2 or np.unique(y_pred[m]).size < 2:
        return float("nan")
    r, _ = spearmanr(y_true[m], y_pred[m])
    return float(r)


def within_one(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m]) <= 1.0 + 1e-8))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    m = np.isfinite(y_true) & np.isfinite(y_prob) & np.isfinite(y_pred)
    yt = y_true[m].astype(int)
    yp = y_pred[m].astype(int)
    pr = y_prob[m]
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)) if len(yt) else float("nan"),
        "f1": float(f1_score(yt, yp, zero_division=0)) if len(yt) else float("nan"),
    }
    if len(np.unique(yt)) > 1:
        out["auc"] = float(roc_auc_score(yt, pr))
    else:
        out["auc"] = float("nan")
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "within_1": within_one(y_true, y_pred),
    }


def ordinal_metrics(y_true: np.ndarray, y_int: np.ndarray, y_exp: np.ndarray) -> dict:
    return {
        "mae": mae(y_true, y_int),
        "spearman": spearman(y_true, y_exp),
        "exact_match": float(
            np.mean(y_true[np.isfinite(y_true)] == y_int[np.isfinite(y_true)])
        )
        if np.any(np.isfinite(y_true))
        else float("nan"),
    }
