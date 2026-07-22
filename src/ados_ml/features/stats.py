"""Weighted aggregation helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def clip01(x: float | None) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    ww = w[mask]
    vv = v[mask]
    return float(np.sum(ww * vv) / np.sum(ww))


def weighted_std(values: Sequence[float], weights: Sequence[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    ww = w[mask]
    vv = v[mask]
    mu = np.sum(ww * vv) / np.sum(ww)
    var = np.sum(ww * (vv - mu) ** 2) / np.sum(ww)
    return float(np.sqrt(max(var, 0.0)))


def weighted_quantile(
    values: Sequence[float], weights: Sequence[float], q: float
) -> float:
    """Weighted quantile in [0, 1] via cumulative weight interpolation."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    vv = v[mask]
    ww = w[mask]
    order = np.argsort(vv)
    vv = vv[order]
    ww = ww[order]
    cw = np.cumsum(ww)
    cutoff = q * cw[-1]
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    idx = min(max(idx, 0), len(vv) - 1)
    return float(vv[idx])


def theil_sen_slope(
    times: Sequence[float], values: Sequence[float], *, max_points: int = 400
) -> float:
    """Median of pairwise slopes (Theil–Sen). Needs >= 2 finite points."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(v)
    t = t[mask]
    v = v[mask]
    n = len(t)
    if n < 2:
        return float("nan")
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        t = t[idx]
        v = v[idx]
        n = len(t)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = t[j] - t[i]
            if abs(dt) < 1e-9:
                continue
            slopes.append((v[j] - v[i]) / dt)
    if not slopes:
        return float("nan")
    return float(np.median(slopes))


def safe_div(num: float, den: float, eps: float = 1e-6) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    return float(num / max(den, eps))
