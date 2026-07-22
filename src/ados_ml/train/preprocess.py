"""Fold-wise imputation and standardization."""

from __future__ import annotations

import numpy as np


class FeaturePreprocessor:
    """Median impute + standardize over (N, T, F), fit on train only."""

    def __init__(self):
        self.medians: np.ndarray | None = None  # (T, F) or (F,) broadcast
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> FeaturePreprocessor:
        # X: (N, T, F)
        n, t, f = X.shape
        flat = X.reshape(-1, f)
        self.medians = np.nanmedian(flat, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        filled = np.where(np.isfinite(X), X, self.medians.reshape(1, 1, f))
        flat2 = filled.reshape(-1, f)
        self.mean = flat2.mean(axis=0)
        self.std = flat2.std(axis=0)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.medians is not None and self.mean is not None and self.std is not None
        f = X.shape[-1]
        filled = np.where(np.isfinite(X), X, self.medians.reshape(1, 1, f))
        return (filled - self.mean.reshape(1, 1, f)) / self.std.reshape(1, 1, f)


class LabelZScorer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y: np.ndarray) -> LabelZScorer:
        y = y[np.isfinite(y)]
        self.mean = float(np.mean(y)) if len(y) else 0.0
        self.std = float(np.std(y)) if len(y) else 1.0
        if self.std < 1e-6:
            self.std = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean
