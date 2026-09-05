"""Task-pair algebra and auxiliary tests for Exp1 multi-task hits.

The result is where the 21 pairs fall in (λ, r) space, not the identity
ρ_z = (ρ_A+ρ_B)/√[2(1+r)] and not H0: ρ_A = ρ_B as a headline. Pair tests
and Cochran Q are auxiliary. See ados_ffm.exp3_findings for the data claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata, t as student_t
from scipy.stats import norm as gauss

from ados_extract.windows import feature_family, is_asr_risky
from ados_ffm.data import FEATURE_COLS, TASKS, TARGETS, apply_target_y
from ados_ffm.exp1_univariate import (
    MIN_N,
    catalog_cols,
    catalog_source,
    merge_task_catalog,
    spearman_rho_fast,
)
from ados_ffm.metrics import bh_fdr

N_BOOT_Q = 9999
N_BOOT_PAIR = 4999
N_BOOT_VERIFY = 2000
SEED_Q = 88001
SEED_PAIR = 89001
SEED_VERIFY = 90001
SEED_CHECK = 87001
RHO_CLIP = 0.999999
FDR_Q = 0.05
MIN_T = 2

TGT_INDEX = {t["name"]: i for i, t in enumerate(TARGETS)}


def fisher_z(rho: float, clip: float = RHO_CLIP) -> float:
    if not np.isfinite(rho):
        return float("nan")
    return float(np.arctanh(np.clip(float(rho), -clip, clip)))


def inv_fisher_z(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(np.tanh(z))


def lambda_star(r: float | np.ndarray) -> float | np.ndarray:
    """Pooling beats the stronger task iff λ > λ*(r).

    r <= -1 is clipped to the boundary value -1 (2*(1+r) clipped to 0) rather
    than treated as undefined, so this also tolerates r slightly past ±1 from
    floating-point noise in resampled correlations. Accepts a scalar or array.
    """
    arr = np.asarray(r, dtype=float)
    out = np.sqrt(np.maximum(0.0, 2.0 * (1.0 + arr))) - 1.0
    return float(out) if out.ndim == 0 else out


def rho_z_identity(rho_a: float, rho_b: float, r: float) -> float:
    """Eq. (3): Pearson of the average of two unit-variance predictors."""
    den = np.sqrt(2.0 * (1.0 + r))
    if not np.isfinite(den) or den <= 0.0:
        return float("nan")
    return float((rho_a + rho_b) / den)


def cochran_q(z: np.ndarray, w: np.ndarray) -> dict[str, float]:
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(z) & np.isfinite(w) & (w > 0.0)
    z, w = z[m], w[m]
    t = int(z.size)
    if t < MIN_T:
        return {"Q": float("nan"), "I2": float("nan"), "zbar": float("nan"), "T": float(t), "W": float("nan")}
    wsum = float(w.sum())
    zbar = float(np.dot(w, z) / wsum)
    q = float(np.dot(w, (z - zbar) ** 2))
    df = t - 1
    if q <= 0.0:
        i2 = 0.0
    else:
        i2 = float(max(0.0, (q - df) / q))
    return {"Q": q, "I2": i2, "zbar": zbar, "T": float(t), "W": wsum}


def ranks_avg(x: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(x, dtype=float), method="average")


def spearman_complete_batch(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman ρ for each row of complete (B, n) arrays."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    rx = rankdata(X, axis=1, method="average")
    ry = rankdata(Y, axis=1, method="average")
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    num = np.sum(rx * ry, axis=1)
    den = np.sqrt(np.sum(rx * rx, axis=1) * np.sum(ry * ry, axis=1))
    out = np.full(X.shape[0], np.nan)
    ok = den > 0.0
    out[ok] = num[ok] / den[ok]
    return out


def spearman_masked_batch(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise Spearman with NaNs omitted. Missing values are ranked as +inf then dropped."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    mask = np.isfinite(X) & np.isfinite(Y)
    n = mask.sum(axis=1)
    xc = np.where(mask, X, np.inf)
    yc = np.where(mask, Y, np.inf)
    rx = rankdata(xc, axis=1, method="average")
    ry = rankdata(yc, axis=1, method="average")
    rx = np.where(mask, rx, np.nan)
    ry = np.where(mask, ry, np.nan)
    rx -= np.nanmean(rx, axis=1, keepdims=True)
    ry -= np.nanmean(ry, axis=1, keepdims=True)
    num = np.nansum(rx * ry, axis=1)
    den = np.sqrt(np.nansum(rx * rx, axis=1) * np.nansum(ry * ry, axis=1))
    out = np.full(X.shape[0], np.nan)
    ok = (n >= 3) & (den > 0.0)
    out[ok] = num[ok] / den[ok]
    return out, n.astype(int)


def hotelling_williams(
    r12: float,
    r13: float,
    r23: float,
    n: int,
) -> dict[str, float]:
    """Williams test of ρ(y, xA) = ρ(y, xB) with dependent correlations. df = n-3."""
    out = {"t": float("nan"), "df": float(max(n - 3, 1)), "p": float("nan"), "detR": float("nan")}
    if n < 5:
        return out
    r12, r13, r23 = float(r12), float(r13), float(r23)
    if not all(np.isfinite(v) for v in (r12, r13, r23)):
        return out
    det = 1.0 - r12**2 - r13**2 - r23**2 + 2.0 * r12 * r13 * r23
    out["detR"] = float(det)
    rbar = 0.5 * (r12 + r13)
    denom = 2.0 * ((n - 1) / (n - 3)) * det + (rbar**2) * (1.0 - r23) ** 3
    if denom <= 0.0 or (1.0 + r23) < 0.0:
        return out
    t_val = (r12 - r13) * np.sqrt((n - 1) * (1.0 + r23) / denom)
    df = n - 3
    p = float(2.0 * student_t.sf(abs(t_val), df))
    out["t"] = float(t_val)
    out["df"] = float(df)
    out["p"] = min(1.0, p)
    return out


def bca_interval(
    obs: float,
    boots: np.ndarray,
    jack: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    if not np.isfinite(obs) or boots.size < 20:
        return float("nan"), float("nan")
    prop = float(np.mean(boots < obs) + 0.5 * np.mean(boots == obs))
    prop = min(max(prop, 1.0 / (boots.size + 1)), 1.0 - 1.0 / (boots.size + 1))
    z0 = float(gauss.ppf(prop))
    jack = np.asarray(jack, dtype=float)
    jack = jack[np.isfinite(jack)]
    a = 0.0
    if jack.size >= 3:
        d = jack.mean() - jack
        den = float(np.sum(d**2))
        if den > 0.0:
            a = float(np.sum(d**3) / (6.0 * den**1.5))
            a = float(np.clip(a, -0.99, 0.99))

    def _adj(p: float) -> float:
        z = float(gauss.ppf(p))
        num = z0 + z
        den = 1.0 - a * num
        if abs(den) < 1e-12:
            return p
        return float(np.clip(gauss.cdf(z0 + num / den), 0.0, 1.0))

    lo = float(np.quantile(boots, _adj(alpha / 2.0)))
    hi = float(np.quantile(boots, _adj(1.0 - alpha / 2.0)))
    return lo, hi


def centered_boot_p(obs: float, boots: np.ndarray) -> float:
    """Two-sided p for H0: θ = 0 from a shifted bootstrap (boots estimate the law of θ)."""
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    if not np.isfinite(obs) or boots.size == 0:
        return float("nan")
    null = boots - obs
    extreme = int(np.sum(np.abs(null) >= abs(obs)))
    return float((1 + extreme) / (1 + boots.size))


def perm_p_right(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return float("nan")
    return float((1 + int(np.sum(null >= obs))) / (1 + null.size))


def orient_lambda(rho_a: float, rho_b: float) -> tuple[float, float, float, int]:
    """Put the larger |ρ| on A and flip so ρ_A > 0. Returns (ρA, ρB, λ, sign_flip)."""
    if not np.isfinite(rho_a) or not np.isfinite(rho_b):
        return float("nan"), float("nan"), float("nan"), 1
    if abs(rho_b) > abs(rho_a):
        rho_a, rho_b = rho_b, rho_a
        swapped = -1
    else:
        swapped = 1
    sgn = 1
    if rho_a < 0.0:
        rho_a, rho_b = -rho_a, -rho_b
        sgn = -1
    if rho_a == 0.0:
        return rho_a, rho_b, float("nan"), sgn * swapped
    return rho_a, rho_b, float(rho_b / rho_a), sgn * swapped


def pool_p1(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Rank-then-pool: z = (rank xA + rank xB) / 2, Pearson(z, rank y)."""
    ra, rb, ry = ranks_avg(x_a), ranks_avg(x_b), ranks_avg(y)
    z = 0.5 * (ra + rb)
    rho_a = spearman_rho_fast(x_a, y)
    rho_b = spearman_rho_fast(x_b, y)
    r = spearman_rho_fast(x_a, x_b)
    zc = z - np.mean(z)
    ryc = ry - np.mean(ry)
    den = float(np.sqrt(np.dot(zc, zc) * np.dot(ryc, ryc)))
    rho_obs = float(np.dot(zc, ryc) / den) if den > 0.0 else float("nan")
    return {
        "rho_a": float(rho_a),
        "rho_b": float(rho_b),
        "r": float(r),
        "rho_z_obs": rho_obs,
        "rho_z_pred": rho_z_identity(rho_a, rho_b, r),
        "n": int(np.size(y)),
    }


def p1_rho_z_batch(Xa: np.ndarray, Xb: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """P1 ρ_z for each row: rank within the row, z = (R_A+R_B)/2, Pearson(z, rank y)."""
    Xa = np.asarray(Xa, dtype=float)
    Xb = np.asarray(Xb, dtype=float)
    Y = np.asarray(Y, dtype=float)
    ra = rankdata(Xa, axis=1, method="average")
    rb = rankdata(Xb, axis=1, method="average")
    z = 0.5 * (ra + rb)
    ry = rankdata(Y, axis=1, method="average")
    zc = z - z.mean(axis=1, keepdims=True)
    ryc = ry - ry.mean(axis=1, keepdims=True)
    num = np.sum(zc * ryc, axis=1)
    den = np.sqrt(np.sum(zc * zc, axis=1) * np.sum(ryc * ryc, axis=1))
    out = np.full(Xa.shape[0], np.nan)
    ok = den > 0.0
    out[ok] = num[ok] / den[ok]
    return out


def percentile_ci(boots: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    if boots.size < 20:
        return float("nan"), float("nan")
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    return bool(np.isfinite(lo) and np.isfinite(hi) and (hi < 0.0 or lo > 0.0))


def _ci_covers_zero(lo: float, hi: float) -> bool:
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi)


def pool_verify_pair(
    xa: np.ndarray,
    xb: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Eq. 3 check on the intersection: point estimates + percentile CIs. No test.

    Ranks are recomputed inside each bootstrap replicate (ties from resampling
    are re-ranked, not frozen from the original sample).
    """
    m = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
    xa, xb, y = xa[m], xb[m], y[m]
    n = int(y.size)
    empty = {
        "n_cap": float(n),
        "rho_a": float("nan"),
        "rho_b": float("nan"),
        "r_spear": float("nan"),
        "rho_z_obs": float("nan"),
        "rho_z_pred": float("nan"),
        "abs_err": float("nan"),
        "rho_a_lo": float("nan"),
        "rho_a_hi": float("nan"),
        "rho_b_lo": float("nan"),
        "rho_b_hi": float("nan"),
        "rho_z_lo": float("nan"),
        "rho_z_hi": float("nan"),
        "rho_a_excludes_0": False,
        "rho_b_excludes_0": False,
        "rho_z_covers_0": False,
        "n_boot": int(n_boot),
        "n_boot_valid": 0,
    }
    if n < MIN_N:
        return empty
    p1 = pool_p1(xa, xb, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    xa_b, xb_b, y_b = xa[idx], xb[idx], y[idx]
    rho_a_b = spearman_complete_batch(xa_b, y_b)
    rho_b_b = spearman_complete_batch(xb_b, y_b)
    rho_z_b = p1_rho_z_batch(xa_b, xb_b, y_b)
    a_lo, a_hi = percentile_ci(rho_a_b)
    b_lo, b_hi = percentile_ci(rho_b_b)
    z_lo, z_hi = percentile_ci(rho_z_b)
    pred = p1["rho_z_pred"]
    obs = p1["rho_z_obs"]
    return {
        "n_cap": float(n),
        "rho_a": float(p1["rho_a"]),
        "rho_b": float(p1["rho_b"]),
        "r_spear": float(p1["r"]),
        "rho_z_obs": float(obs),
        "rho_z_pred": float(pred),
        "abs_err": float(abs(obs - pred)) if np.isfinite(obs) and np.isfinite(pred) else float("nan"),
        "rho_a_lo": a_lo,
        "rho_a_hi": a_hi,
        "rho_b_lo": b_lo,
        "rho_b_hi": b_hi,
        "rho_z_lo": z_lo,
        "rho_z_hi": z_hi,
        "rho_a_excludes_0": _ci_excludes_zero(a_lo, a_hi),
        "rho_b_excludes_0": _ci_excludes_zero(b_lo, b_hi),
        "rho_z_covers_0": _ci_covers_zero(z_lo, z_hi),
        "n_boot": int(n_boot),
        "n_boot_valid": int(np.isfinite(rho_z_b).sum()),
    }


def pool_p2(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pool-then-rank: z-score raw x, average, then Spearman with y."""
    def _z(v: np.ndarray) -> np.ndarray:
        s = float(np.std(v, ddof=0))
        if s <= 0.0:
            return np.zeros_like(v, dtype=float)
        return (v - float(np.mean(v))) / s

    z = 0.5 * (_z(x_a) + _z(x_b))
    rho_a = float(np.corrcoef(x_a, y)[0, 1]) if np.std(x_a) > 0 and np.std(y) > 0 else float("nan")
    rho_b = float(np.corrcoef(x_b, y)[0, 1]) if np.std(x_b) > 0 and np.std(y) > 0 else float("nan")
    r = float(np.corrcoef(x_a, x_b)[0, 1]) if np.std(x_a) > 0 and np.std(x_b) > 0 else float("nan")
    return {
        "rho_a": rho_a,
        "rho_b": rho_b,
        "r": r,
        "rho_z_obs": spearman_rho_fast(z, y),
        "rho_z_pred": rho_z_identity(rho_a, rho_b, r),
        "n": int(np.size(y)),
    }


@dataclass
class Panel:
    pids: np.ndarray
    y: dict[str, np.ndarray]
    X: dict[str, dict[int, np.ndarray]]
    features: list[str]
    tasks: tuple[int, ...]
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = int(self.pids.size)


def build_panel(
    dyn: pd.DataFrame,
    new: pd.DataFrame,
    lab: pd.DataFrame,
    cohort: list[str],
    tasks: tuple[int, ...] = TASKS,
) -> Panel:
    pids = np.asarray([str(p) for p in cohort], dtype=object)
    loc = {str(p): i for i, p in enumerate(pids)}
    n = int(pids.size)
    y: dict[str, np.ndarray] = {}
    for tgt in TARGETS:
        raw = apply_target_y(lab.reindex(pids)[tgt["column"]])
        y[tgt["name"]] = np.asarray(raw.to_numpy(), dtype=float)
    tabs = {tid: merge_task_catalog(dyn, new, int(tid), list(cohort)) for tid in tasks}
    names: set[str] = set()
    for tab in tabs.values():
        names.update(catalog_cols(list(tab.columns)))
    features = sorted(names)
    X: dict[str, dict[int, np.ndarray]] = {f: {} for f in features}
    for tid, tab in tabs.items():
        if tab.empty:
            for f in features:
                X[f][int(tid)] = np.full(n, np.nan)
            continue
        pos = tab["participant_id"].astype(str).map(loc)
        for f in features:
            arr = np.full(n, np.nan)
            if f in tab.columns:
                vals = pd.to_numeric(tab[f], errors="coerce").to_numpy(dtype=float)
                for i, v in zip(pos.to_numpy(), vals):
                    if pd.notna(i):
                        arr[int(i)] = v
            X[f][int(tid)] = arr
    return Panel(pids=pids, y=y, X=X, features=features, tasks=tuple(int(t) for t in tasks))


def task_point(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < MIN_N:
        return {"n": float(n), "rho": float("nan"), "z": float("nan"), "w": float("nan")}
    xv, yv = x[m], y[m]
    if float(np.std(xv)) <= 0.0 or float(np.std(yv)) <= 0.0:
        return {"n": float(n), "rho": float("nan"), "z": float("nan"), "w": float("nan")}
    rho = spearman_rho_fast(xv, yv)
    z = fisher_z(rho)
    w = float(n - 3) if n > 3 else float("nan")
    return {"n": float(n), "rho": float(rho), "z": z, "w": w}


def observed_tasks(x_by_task: dict[int, np.ndarray], y: np.ndarray) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for tid, x in x_by_task.items():
        pt = task_point(x, y)
        if np.isfinite(pt["z"]) and np.isfinite(pt["w"]) and pt["w"] > 0.0:
            out[int(tid)] = pt
    return out


def q_observed(task_pts: dict[int, dict[str, float]]) -> dict[str, float]:
    tids = sorted(task_pts)
    z = np.array([task_pts[t]["z"] for t in tids], dtype=float)
    w = np.array([task_pts[t]["w"] for t in tids], dtype=float)
    q = cochran_q(z, w)
    rhos = np.array([task_pts[t]["rho"] for t in tids], dtype=float)
    ns = np.array([task_pts[t]["n"] for t in tids], dtype=float)
    q["rho_min"] = float(np.min(rhos)) if rhos.size else float("nan")
    q["rho_max"] = float(np.max(rhos)) if rhos.size else float("nan")
    q["n_mean"] = float(np.mean(ns)) if ns.size else float("nan")
    q["n_min"] = float(np.min(ns)) if ns.size else float("nan")
    q["sign_pos"] = int(np.sum(rhos > 0.0))
    q["sign_neg"] = int(np.sum(rhos < 0.0))
    q["n_sign_discord"] = int((rhos.min() < 0.0) and (rhos.max() > 0.0)) if rhos.size else 0
    df = float(q["T"] - 1.0) if np.isfinite(q["T"]) else float("nan")
    q["p_chi2"] = float(chi2.sf(q["Q"], df)) if np.isfinite(q["Q"]) and df > 0 else float("nan")
    return q


def bootstrap_q(
    x_by_task: dict[int, np.ndarray],
    y: np.ndarray,
    task_pts: dict[int, dict[str, float]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    q_hat = q_observed(task_pts)
    tids = sorted(task_pts)
    z_hat = np.array([task_pts[t]["z"] for t in tids], dtype=float)
    w = np.array([task_pts[t]["w"] for t in tids], dtype=float)
    zbar = float(q_hat["zbar"])
    n = int(y.size)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    z_b = np.full((n_boot, len(tids)), np.nan)
    yb = y[idx]
    for j, tid in enumerate(tids):
        xb = x_by_task[tid][idx]
        rho, nb = spearman_masked_batch(xb, yb)
        ok = (nb >= MIN_N) & np.isfinite(rho)
        z_b[ok, j] = np.arctanh(np.clip(rho[ok], -RHO_CLIP, RHO_CLIP))
    valid = np.all(np.isfinite(z_b), axis=1)
    z_use = z_b[valid]
    if z_use.size == 0:
        return {**q_hat, "p": float("nan"), "n_boot_valid": 0, "n_boot": int(n_boot)}
    z_tilde = z_use - z_hat.reshape(1, -1) + zbar
    wsum = float(w.sum())
    zbar_b = (z_tilde * w.reshape(1, -1)).sum(axis=1) / wsum
    q_null = ((z_tilde - zbar_b.reshape(-1, 1)) ** 2 * w.reshape(1, -1)).sum(axis=1)
    p = perm_p_right(float(q_hat["Q"]), q_null)
    return {
        **q_hat,
        "p": p,
        "n_boot_valid": int(z_use.shape[0]),
        "n_boot": int(n_boot),
        "Q_null_med": float(np.median(q_null)),
        "Q_null_p95": float(np.percentile(q_null, 95)),
    }


def _delta_z_one(xa: np.ndarray, xb: np.ndarray, y: np.ndarray) -> float:
    ra = spearman_rho_fast(xa, y)
    rb = spearman_rho_fast(xb, y)
    if not np.isfinite(ra) or not np.isfinite(rb):
        return float("nan")
    return fisher_z(ra) - fisher_z(rb)


def pair_contrast(
    xa: np.ndarray,
    xb: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    m = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
    xa, xb, y = xa[m], xb[m], y[m]
    n = int(y.size)
    empty = {
        "n_cap": float(n),
        "rho_a": float("nan"),
        "rho_b": float("nan"),
        "r_spear": float("nan"),
        "r_pear": float("nan"),
        "delta_rho": float("nan"),
        "delta_z": float("nan"),
        "p_boot": float("nan"),
        "p_hw": float("nan"),
        "t_hw": float("nan"),
        "ci_z_lo": float("nan"),
        "ci_z_hi": float("nan"),
        "ci_rho_lo": float("nan"),
        "ci_rho_hi": float("nan"),
        "n_boot_valid": 0,
    }
    if n < MIN_N:
        return empty
    rho_a = spearman_rho_fast(xa, y)
    rho_b = spearman_rho_fast(xb, y)
    r_s = spearman_rho_fast(xa, xb)
    r_p = float(np.corrcoef(xa, xb)[0, 1]) if np.std(xa) > 0 and np.std(xb) > 0 else float("nan")
    dz = fisher_z(rho_a) - fisher_z(rho_b)
    hw = hotelling_williams(rho_a, rho_b, r_s, n)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ya = y[idx]
    ra = spearman_complete_batch(xa[idx], ya)
    rb = spearman_complete_batch(xb[idx], ya)
    ok = np.isfinite(ra) & np.isfinite(rb)
    dz_b = np.arctanh(np.clip(ra[ok], -RHO_CLIP, RHO_CLIP)) - np.arctanh(
        np.clip(rb[ok], -RHO_CLIP, RHO_CLIP)
    )
    dr_b = ra[ok] - rb[ok]
    jack = np.empty(n, dtype=float)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        jack[i] = _delta_z_one(xa[keep], xb[keep], y[keep])
        keep[i] = True
    jack_r = np.empty(n, dtype=float)
    keep[:] = True
    for i in range(n):
        keep[i] = False
        jack_r[i] = spearman_rho_fast(xa[keep], y[keep]) - spearman_rho_fast(xb[keep], y[keep])
        keep[i] = True
    lo, hi = bca_interval(dz, dz_b, jack)
    lo_r, hi_r = bca_interval(float(rho_a - rho_b), dr_b, jack_r)
    return {
        "n_cap": float(n),
        "rho_a": float(rho_a),
        "rho_b": float(rho_b),
        "r_spear": float(r_s),
        "r_pear": float(r_p),
        "delta_rho": float(rho_a - rho_b),
        "delta_z": float(dz),
        "p_boot": centered_boot_p(dz, dz_b),
        "p_hw": float(hw["p"]),
        "t_hw": float(hw["t"]),
        "detR": float(hw["detR"]),
        "ci_z_lo": lo,
        "ci_z_hi": hi,
        "ci_rho_lo": lo_r,
        "ci_rho_hi": hi_r,
        "n_boot_valid": int(dz_b.size),
    }


def maxstat_p(
    x_by_task: dict[int, np.ndarray],
    y: np.ndarray,
    hit_tids: list[int],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """p for max |Δz| over Exp1-hit task pairs. Fixes selection of the largest |Δρ|."""
    tids = sorted({int(t) for t in hit_tids})
    cand = [(a, b) for i, a in enumerate(tids) for b in tids[i + 1 :]]
    empty = {"n_hit_tasks": float(len(tids)), "n_cand_pairs": float(len(cand)), "max_abs_dz": float("nan"), "p_maxstat": float("nan"), "n_boot_valid": 0}
    if len(cand) == 0:
        return empty
    used: list[tuple[int, int]] = []
    obs_dz: list[float] = []
    for a, b in cand:
        xa, xb = x_by_task[a], x_by_task[b]
        m = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
        if int(m.sum()) < MIN_N:
            continue
        dz = fisher_z(spearman_rho_fast(xa[m], y[m])) - fisher_z(spearman_rho_fast(xb[m], y[m]))
        if np.isfinite(dz):
            used.append((a, b))
            obs_dz.append(float(dz))
    if not obs_dz:
        return empty
    obs_max = float(max(abs(d) for d in obs_dz))
    n = int(y.size)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    yb = y[idx]
    mx = np.full(n_boot, np.nan)
    for (a, b), dz_hat in zip(used, obs_dz):
        ra, na = spearman_masked_batch(x_by_task[a][idx], yb)
        rb, nb = spearman_masked_batch(x_by_task[b][idx], yb)
        ok = (na >= MIN_N) & (nb >= MIN_N) & np.isfinite(ra) & np.isfinite(rb)
        cent = np.full(n_boot, np.nan)
        raw = np.arctanh(np.clip(ra[ok], -RHO_CLIP, RHO_CLIP)) - np.arctanh(
            np.clip(rb[ok], -RHO_CLIP, RHO_CLIP)
        )
        cent[ok] = np.abs(raw - dz_hat)
        mx = np.fmax(mx, cent)
    valid = np.isfinite(mx)
    p = float((1 + int(np.sum(mx[valid] >= obs_max))) / (1 + int(valid.sum()))) if valid.any() else float("nan")
    return {
        "n_hit_tasks": float(len(tids)),
        "n_cand_pairs": float(len(cand)),
        "n_obs_pairs": float(len(obs_dz)),
        "max_abs_dz": obs_max,
        "p_maxstat": p,
        "n_boot_valid": int(valid.sum()),
    }


def attach_bh_one(df: pd.DataFrame, p_col: str = "p_boot") -> pd.DataFrame:
    """BH on the whole table as one family (the 21 Exp1 multi-task pairs)."""
    out = df.copy()
    q = bh_fdr(out[p_col].to_numpy(dtype=float))
    out["q_fdr"] = q
    out["m"] = int(np.isfinite(out[p_col].to_numpy(dtype=float)).sum())
    out["fdr_sig"] = out["q_fdr"] < FDR_Q
    return out


def attach_bh_global(df: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    out = df.copy()
    q = bh_fdr(out[p_col].to_numpy(dtype=float))
    out["q_fdr"] = q
    out["m"] = int(np.isfinite(out[p_col].to_numpy(dtype=float)).sum())
    out["fdr_sig"] = out["q_fdr"] < FDR_Q
    return out


def family_seed(feature: str, target: str, base: int) -> int:
    h = 0
    for ch in str(feature):
        h = (h * 131 + ord(ch)) % 1_000_003
    return int(base + h * 3 + TGT_INDEX[target])


def feature_meta(name: str) -> dict[str, Any]:
    return {
        "source": catalog_source(name),
        "family": "frozen34" if name in FEATURE_COLS else feature_family(name),
        "asr_risky": bool(is_asr_risky(name)),
    }


def exp1_multitask_pairs(hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feat, tgt), g in hits.groupby(["feature", "target"]):
        if g["task"].nunique() < 2:
            continue
        recs = list(
            zip(
                g["task"].astype(int).tolist(),
                g["task_ja"].tolist(),
                g["rho"].to_numpy(dtype=float),
                g["n"].to_numpy(dtype=float),
            )
        )
        signs = np.sign([r[2] for r in recs])
        rows.append(
            {
                "feature": feat,
                "target": tgt,
                "n_hit_tasks": int(g["task"].nunique()),
                "sign_flip": bool((signs.min() < 0) and (signs.max() > 0)),
                "tasks": recs,
            }
        )
    return pd.DataFrame(rows)


def pick_figure_pair(task_recs: list[tuple], panel: Panel, feature: str, target: str) -> dict | None:
    """Among Exp1-hit tasks, pick the intersection pair with largest |Δρ|."""
    y = panel.y[target]
    xmap = panel.X[feature]
    best = None
    best_abs = -1.0
    tids = [int(t[0]) for t in task_recs]
    rho_hit = {int(t[0]): float(t[2]) for t in task_recs}
    n_hit = {int(t[0]): float(t[3]) for t in task_recs}
    ja = {int(t[0]): str(t[1]) for t in task_recs}
    for i, ta in enumerate(tids):
        for tb in tids[i + 1 :]:
            xa, xb = xmap[ta], xmap[tb]
            m = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
            if int(m.sum()) < MIN_N:
                continue
            ra = spearman_rho_fast(xa[m], y[m])
            rb = spearman_rho_fast(xb[m], y[m])
            d = abs(ra - rb)
            if d > best_abs:
                best_abs = d
                best = (ta, tb, ra, rb, xa[m].copy(), xb[m].copy(), y[m].copy())
    if best is None:
        return None
    ta, tb, ra, rb, xa, xb, yy = best
    p1 = pool_p1(xa, xb, yy)
    p2 = pool_p2(xa, xb, yy)
    # Orient: larger |ρ| first, then flip feature so ρ_A > 0.
    if abs(p1["rho_b"]) > abs(p1["rho_a"]):
        ta, tb = tb, ta
        xa, xb = xb, xa
        p1 = pool_p1(xa, xb, yy)
        p2 = pool_p2(xa, xb, yy)
    rho_a_o, rho_b_o, lam, _ = orient_lambda(p1["rho_a"], p1["rho_b"])
    r = p1["r"]
    return {
        "feature": feature,
        "target": target,
        **feature_meta(feature),
        "task_a": int(ta),
        "task_a_ja": ja[ta],
        "task_b": int(tb),
        "task_b_ja": ja[tb],
        "n_hit_a": n_hit[ta],
        "n_hit_b": n_hit[tb],
        "rho_hit_a": rho_hit[ta],
        "rho_hit_b": rho_hit[tb],
        "n_cap": p1["n"],
        "rho_cap_a": p1["rho_a"],
        "rho_cap_b": p1["rho_b"],
        "r_spear": p1["r"],
        "rho_a_orient": rho_a_o,
        "rho_b_orient": rho_b_o,
        "lambda": lam,
        "lambda_star": lambda_star(r),
        "pool_worse": bool(np.isfinite(lam) and np.isfinite(lambda_star(r)) and lam <= lambda_star(r)),
        "p1_rho_z_obs": p1["rho_z_obs"],
        "p1_rho_z_pred": p1["rho_z_pred"],
        "p2_rho_z_obs": p2["rho_z_obs"],
        "p2_rho_z_pred": p2["rho_z_pred"],
        "p2_r_pear": p2["r"],
        "sign_flip_cap": bool(p1["rho_a"] * p1["rho_b"] < 0.0),
    }


def verify_signflip_table(
    panel: Panel,
    flips: pd.DataFrame,
    *,
    n_boot: int,
) -> pd.DataFrame:
    """Eq. 3 verification on the sign-flip combos. Same n∩ as the pair tests."""
    rows = []
    for rec in flips.itertuples(index=False):
        feat = str(rec.feature)
        tgt = str(rec.target)
        ta = int(rec.task_pos)
        tb = int(rec.task_neg)
        got = pool_verify_pair(
            panel.X[feat][ta],
            panel.X[feat][tb],
            panel.y[tgt],
            n_boot=n_boot,
            seed=family_seed(feat, tgt, SEED_VERIFY) + ta * 17 + tb,
        )
        exp_a = float(getattr(rec, "rho_cap_pos", np.nan))
        exp_b = float(getattr(rec, "rho_cap_neg", np.nan))
        exp_r = float(getattr(rec, "r_spear", np.nan))
        match = True
        if np.isfinite(exp_a):
            match = match and abs(got["rho_a"] - exp_a) < 1e-9
        if np.isfinite(exp_b):
            match = match and abs(got["rho_b"] - exp_b) < 1e-9
        if np.isfinite(exp_r):
            match = match and abs(got["r_spear"] - exp_r) < 1e-9
        rows.append(
            {
                "feature": feat,
                "target": tgt,
                **feature_meta(feat),
                "task_pos": ta,
                "task_pos_ja": rec.task_pos_ja,
                "task_neg": tb,
                "task_neg_ja": rec.task_neg_ja,
                "n_cap": got["n_cap"],
                "rho_pos": got["rho_a"],
                "rho_pos_lo": got["rho_a_lo"],
                "rho_pos_hi": got["rho_a_hi"],
                "rho_pos_excludes_0": got["rho_a_excludes_0"],
                "rho_neg": got["rho_b"],
                "rho_neg_lo": got["rho_b_lo"],
                "rho_neg_hi": got["rho_b_hi"],
                "rho_neg_excludes_0": got["rho_b_excludes_0"],
                "r_spear": got["r_spear"],
                "rho_z_pred": got["rho_z_pred"],
                "rho_z_obs": got["rho_z_obs"],
                "rho_z_lo": got["rho_z_lo"],
                "rho_z_hi": got["rho_z_hi"],
                "rho_z_covers_0": got["rho_z_covers_0"],
                "abs_err": got["abs_err"],
                "match_cap": bool(match),
                "n_boot": got["n_boot"],
                "n_boot_valid": got["n_boot_valid"],
            }
        )
    return pd.DataFrame(rows)


def pool_verify_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n_pairs": 0}
    err = df["abs_err"].to_numpy(dtype=float)
    pred = df["rho_z_pred"].to_numpy(dtype=float)
    obs = df["rho_z_obs"].to_numpy(dtype=float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    corr = float("nan")
    if int(ok.sum()) >= 3 and float(np.std(pred[ok])) > 0.0 and float(np.std(obs[ok])) > 0.0:
        corr = float(np.corrcoef(pred[ok], obs[ok])[0, 1])
    return {
        "n_pairs": int(len(df)),
        "median_abs_err": float(np.nanmedian(err)),
        "max_abs_err": float(np.nanmax(err)),
        "corr_pred_obs": corr,
        "n_rho_z_covers_0": int(df["rho_z_covers_0"].sum()),
        "n_rho_pos_excludes_0": int(df["rho_pos_excludes_0"].sum()),
        "n_rho_neg_excludes_0": int(df["rho_neg_excludes_0"].sum()),
        "n_both_sides_exclude_0": int(
            (df["rho_pos_excludes_0"] & df["rho_neg_excludes_0"]).sum()
        ),
        "n_pos_excludes_and_z_covers": int(
            (df["rho_pos_excludes_0"] & df["rho_z_covers_0"]).sum()
        ),
        "n_match_cap": int(df["match_cap"].sum()) if "match_cap" in df.columns else None,
    }


def plot_eq3_verify(df: pd.DataFrame, path: Path) -> None:
    """CI forest only. The identity scatter is algebra, not a result."""
    from ados_ffm.exp3_findings import plot_ci_forest

    plot_ci_forest(df, path)


def self_check() -> list[str]:
    """Synthetic sanity checks. Returns a list of failure messages (empty = pass)."""
    err: list[str] = []
    rng = np.random.default_rng(SEED_CHECK)
    n = 60
    y = rng.normal(size=n)
    xa = y + 0.4 * rng.normal(size=n)
    xb = 0.5 * xa + 0.5 * rng.normal(size=n)
    p1 = pool_p1(xa, xb, y)
    if abs(p1["rho_z_obs"] - p1["rho_z_pred"]) > 0.03:
        err.append(f"P1 identity drift {p1['rho_z_obs']:.4f} vs {p1['rho_z_pred']:.4f}")
    xb_flip = -y + 0.45 * rng.normal(size=n)
    xa_flip = y + 0.45 * rng.normal(size=n)
    v = pool_verify_pair(xa_flip, xb_flip, y, n_boot=400, seed=SEED_CHECK + 3)
    if abs(v["rho_z_obs"] - v["rho_z_pred"]) > 0.03:
        err.append(f"verify identity drift {v['rho_z_obs']:.4f} vs {v['rho_z_pred']:.4f}")
    if abs(v["rho_z_pred"]) > 0.18:
        err.append(f"flip-toy pred should be near 0, got {v['rho_z_pred']:.4f}")
    if not v["rho_z_covers_0"]:
        err.append(f"flip-toy ρ_z CI should cover 0, got [{v['rho_z_lo']:.3f},{v['rho_z_hi']:.3f}]")
    if not v["rho_a_excludes_0"]:
        err.append(f"flip-toy ρ_A CI should exclude 0, got [{v['rho_a_lo']:.3f},{v['rho_a_hi']:.3f}]")
    hw0 = hotelling_williams(0.3, 0.3, 0.2, n)
    if abs(hw0["t"]) > 1e-9:
        err.append(f"HW t should be 0 when ρ equal, got {hw0['t']}")
    hw1 = hotelling_williams(0.336, -0.25, 0.0, 51)
    if not (hw1["p"] < 0.05):
        err.append(f"HW should reject sign-flip toy, p={hw1['p']}")

    T = 6
    y = rng.normal(size=80)
    x_h = {t: 0.35 * y + rng.normal(size=80) for t in range(T)}
    pts_h = observed_tasks(x_h, y)
    qh = bootstrap_q(x_h, y, pts_h, n_boot=400, seed=SEED_CHECK)
    x_a = {}
    for t in range(T):
        sgn = 1.0 if t < T / 2 else -1.0
        x_a[t] = sgn * 0.55 * y + 0.35 * rng.normal(size=80)
    pts_a = observed_tasks(x_a, y)
    qa = bootstrap_q(x_a, y, pts_a, n_boot=400, seed=SEED_CHECK + 1)
    if not (qa["p"] < 0.05):
        err.append(f"Q failed to reject heterogeneous toy p={qa['p']}")
    if qa["p"] >= qh["p"]:
        err.append(f"Q H1 p={qa['p']} not < H0 p={qh['p']}")
    return err


__all__ = [
    "N_BOOT_PAIR",
    "N_BOOT_Q",
    "N_BOOT_VERIFY",
    "Panel",
    "attach_bh_global",
    "attach_bh_one",
    "bootstrap_q",
    "build_panel",
    "exp1_multitask_pairs",
    "family_seed",
    "feature_meta",
    "maxstat_p",
    "observed_tasks",
    "pair_contrast",
    "pick_figure_pair",
    "plot_eq3_verify",
    "pool_verify_pair",
    "pool_verify_summary",
    "q_observed",
    "self_check",
    "verify_signflip_table",
]
