"""Participant-level CV splits."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def sa_tertiles(sa: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Integer stratum labels from SA quantiles (training-agnostic on full cohort)."""
    # Use rank-based bins so ties don't break StratifiedKFold
    order = sa.argsort()
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(sa))
    return np.minimum(n_bins - 1, ranks * n_bins // max(len(sa), 1))


def make_folds(
    n: int,
    *,
    n_folds: int,
    stratify: np.ndarray | None,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if stratify is None:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, n_folds)
        out = []
        for i in range(n_folds):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            out.append((tr, te))
        return out

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # StratifiedKFold needs at least n_folds samples per class — merge rare bins
    y = np.asarray(stratify).copy()
    # If any class has < 2, fall back to non-stratified
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return make_folds(n, n_folds=n_folds, stratify=None, seed=seed)
    return list(skf.split(np.zeros(n), y))
