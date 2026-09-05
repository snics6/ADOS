"""Ridge CV helpers used by the Exp2 Ridge auxiliary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from ados_ffm.data import ALPHAS
from ados_ffm.exp1_univariate import spearman_rho_fast

SOURCES: tuple[str, ...] = ("child", "examiner", "dyad")
MIN_N = 12
N_PERM = 1000
SEED_CV = 51001
SEED_PERM = 52001


def cv_schedule(n: int) -> tuple[int, int] | None:
    if n >= 40:
        return 5, 10
    if n >= 28:
        return 4, 10
    if n >= 18:
        return 3, 15
    if n >= 12:
        return 2, 20
    return None


def strata_exp2(y: np.ndarray, group: np.ndarray, n_splits: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    group = np.asarray(group)
    try:
        edges = np.unique(np.quantile(y, [0.25, 0.5, 0.75]))
        base = np.digitize(y, edges) if edges.size else np.zeros(len(y), dtype=int)
    except ValueError:
        base = np.zeros(len(y), dtype=int)
    g = np.array([1 if str(v) == "preterm" else 0 for v in group], dtype=int)
    combined = base * 2 + g
    _, cnt = np.unique(combined, return_counts=True)
    if cnt.size and int(cnt.min()) >= n_splits:
        return combined
    _, cnt = np.unique(base, return_counts=True)
    if cnt.size and int(cnt.min()) >= n_splits:
        return base
    two = (y >= np.median(y)).astype(int)
    _, cnt = np.unique(two, return_counts=True)
    if cnt.size and int(cnt.min()) >= n_splits:
        return two
    return np.zeros(len(y), dtype=int)


def _splits(n: int, y: np.ndarray, group: np.ndarray, n_splits: int, n_repeats: int, seed: int):
    dummy = np.zeros((n, 1))
    st = strata_exp2(y, group, n_splits)
    try:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        return list(cv.split(dummy, st))
    except ValueError:
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        return list(cv.split(dummy))


def score_summary(scores: list[float]) -> dict[str, float]:
    a = np.asarray(scores, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "rho": float("nan"),
            "rho_sd": float("nan"),
            "rho_pos_frac": float("nan"),
            "rho_p025": float("nan"),
            "rho_p975": float("nan"),
        }
    return {
        "rho": float(np.mean(a)),
        "rho_sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "rho_pos_frac": float(np.mean(a > 0)),
        "rho_p025": float(np.quantile(a, 0.025)),
        "rho_p975": float(np.quantile(a, 0.975)),
    }


def run_ridge_cols_cv(
    frame: pd.DataFrame,
    cols: list[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> dict[str, Any]:
    y = frame["y"].to_numpy(dtype=float)
    group = frame["group"].to_numpy()
    splits = _splits(len(y), y, group, n_splits, n_repeats, seed)
    X = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scores: list[float] = []
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        pred = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr]).predict(
            sc.transform(X[te])
        )
        yte = y[te]
        if float(np.std(pred)) <= 1e-12 or float(np.std(yte)) <= 1e-12:
            scores.append(0.0)
        else:
            scores.append(spearman_rho_fast(yte, pred))
    n_folds = max(len(splits), 1)
    return {**score_summary(scores), "n_folds": n_folds}


def perm_p(observed: float, null: np.ndarray, *, side: str = "greater") -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return float("nan")
    if side == "greater":
        k = int(np.sum(null >= float(observed)))
    else:
        k = int(np.sum(null <= float(observed)))
    return float((1 + k) / (1 + null.size))


def perm_cell_ridge(
    frame: pd.DataFrame,
    cols: list[str],
    n_splits: int,
    n_repeats: int,
    n_perm: int,
    seed: int,
    obs_rho: float,
) -> float:
    y = frame["y"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    null = np.full(n_perm, np.nan)
    for i in range(n_perm):
        sh = frame.copy()
        sh["y"] = rng.permutation(y)
        null[i] = run_ridge_cols_cv(sh, cols, n_splits, n_repeats, seed + 1 + i)["rho"]
    return perm_p(obs_rho, null, side="greater")
