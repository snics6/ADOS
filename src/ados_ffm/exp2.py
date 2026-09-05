"""Experiment 2 (main): LOPO with Exp1-like Stage A on train, k=1, sign(ρ)·x.

For each Exp1 FDR cell (task × target × source):
  - Feature pool = all catalog features for that source
  - Each LOPO fold: Stage A (S >= locked τ) on train, pick max |Spearman|
    among passers (else fallback max |ρ|); predict left-out with sign(train ρ)·x
  - Outer y-permutation re-runs selection inside every LOPO fold
  - BH within (task × score)

Auxiliary (Exp1 hits as-is, Ridge): `ados_ffm.exp2_ridge`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ados_ffm.data import TASK_JA, apply_target_y
from ados_ffm.exp1_univariate import (
    catalog_cols,
    catalog_source,
    half_stability,
    merge_task_catalog,
    spearman_rho_fast,
)
from ados_ffm.kinds import feature_kind
from ados_ffm.metrics import bh_fdr
from ados_ffm.ridge_cv import MIN_N, N_PERM, SEED_CV, SEED_PERM, SOURCES

HIT_PATH_DEFAULT = "outputs/exp1/hits_main.csv"
THRESH_PATH_DEFAULT = "outputs/exp1/thresholds.json"
N_BOOT = 40
SEED_LOPO = 77001


def attach_fdr_task_target(df: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    """BH within (task × score). m = sources that actually ran (max 3)."""
    out = df.copy()
    out["q_fdr"] = np.nan
    out["m"] = 0
    out["q_family"] = ""
    for (task, target), g in out.groupby(["task", "target"], dropna=False):
        idx = g.index
        p = g[p_col].to_numpy(dtype=float)
        q = bh_fdr(p)
        m = int(np.isfinite(p).sum())
        out.loc[idx, "q_fdr"] = q
        out.loc[idx, "m"] = m
        out.loc[idx, "q_family"] = f"{int(task)}_{target}"
    out["fdr_sig"] = out["q_fdr"] < 0.05
    return out


def load_thresholds(path: Path | str = THRESH_PATH_DEFAULT) -> dict[str, dict[str, float]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw["thresholds"]


def cell_seed(task: int, source: str, target: str) -> int:
    return (
        SEED_LOPO
        + int(task) * 1009
        + SOURCES.index(source) * 17
        + (0 if target == "SA" else 1 if target == "RRB" else 2)
    )


def select_one(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    seed: int,
    *,
    n_boot: int = N_BOOT,
) -> tuple[int | None, str, float, int]:
    """Stage A: S>=τ; then max |ρ| among passers (else global max |ρ|)."""
    n_f = X.shape[1]
    S = np.full(n_f, np.nan)
    R = np.full(n_f, np.nan)
    for j in range(n_f):
        x = X[:, j]
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < MIN_N:
            continue
        xx, yy = x[m], y[m]
        R[j] = spearman_rho_fast(xx, yy)
        S[j] = half_stability(
            xx,
            yy,
            n_rep=n_boot,
            rng=np.random.default_rng(seed + 17 * j + 3),
            positive_only=False,
        )
    pass_idx = [
        j
        for j in range(n_f)
        if np.isfinite(S[j]) and S[j] >= tau and np.isfinite(R[j])
    ]
    if pass_idx:
        j = max(pass_idx, key=lambda jj: abs(R[jj]))
        return j, "stage", float(R[j]), len(pass_idx)
    ok = [j for j in range(n_f) if np.isfinite(R[j])]
    if not ok:
        return None, "none", float("nan"), 0
    j = max(ok, key=lambda jj: abs(R[jj]))
    return j, "fallback", float(R[j]), 0


def lopo_signx(
    X: np.ndarray,
    y: np.ndarray,
    cols: list[str],
    tau: float,
    seed_base: int,
    *,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    n = len(y)
    oof = np.full(n, np.nan)
    picks: list[str] = []
    modes: list[str] = []
    npass_list: list[int] = []
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        j, mode, _rj, npass = select_one(
            X[tr], y[tr], tau, seed_base + i * 131, n_boot=n_boot
        )
        modes.append(mode)
        npass_list.append(npass)
        if j is None:
            picks.append("")
            continue
        picks.append(cols[j])
        if not np.isfinite(X[i, j]):
            continue
        r = spearman_rho_fast(X[tr, j], y[tr])
        if not np.isfinite(r):
            continue
        s = 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)
        oof[i] = s * float(X[i, j])
    m = np.isfinite(oof)
    rho = spearman_rho_fast(y[m], oof[m]) if int(m.sum()) >= 8 else float("nan")
    top = Counter([p for p in picks if p]).most_common(3)
    return {
        "rho": float(rho) if np.isfinite(rho) else float("nan"),
        "n": int(n),
        "n_oof": int(m.sum()),
        "picks": picks,
        "modes": modes,
        "npass_mean": float(np.mean(npass_list)) if npass_list else 0.0,
        "stage_frac": float(np.mean([m == "stage" for m in modes])) if modes else 0.0,
        "top": top,
    }


def prepare_jobs(
    hits: pd.DataFrame,
    dyn: pd.DataFrame,
    new: pd.DataFrame,
    lab: pd.DataFrame,
    cohort: list[str],
    targets: tuple[dict[str, Any], ...],
    task_ids: tuple[int, ...],
    sources: tuple[str, ...] = SOURCES,
    *,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One job per Exp1 FDR cell (task × target × source) with source feature pool."""
    thr = thresholds if thresholds is not None else load_thresholds()
    tgt_of = {t["name"]: t for t in targets}
    cells = (
        hits.groupby(["task", "target", "source"], as_index=False)
        .size()
        .rename(columns={"size": "n_exp1_hits"})
    )
    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    packed: dict[int, tuple[pd.DataFrame, list[str]]] = {}

    for task in sorted(set(int(t) for t in cells["task"]) | set(task_ids)):
        if task not in task_ids:
            continue
        frame = merge_task_catalog(dyn, new, int(task), cohort)
        packed[int(task)] = (frame, catalog_cols(list(frame.columns)))

    for r in cells.itertuples(index=False):
        task = int(r.task)
        if task not in task_ids or task not in packed:
            continue
        source = str(r.source)
        target = str(r.target)
        if source not in sources or target not in tgt_of:
            continue
        frame, cols_all = packed[task]
        cols = [c for c in cols_all if catalog_source(c) == source]
        base = {
            "task": task,
            "task_ja": TASK_JA[task],
            "source": source,
            "target": target,
            "n_exp1_hits": int(r.n_exp1_hits),
        }
        if not cols:
            skipped.append({**base, "reason": "no_pool_cols"})
            continue
        pids = frame["participant_id"].astype(str)
        y = apply_target_y(lab.reindex(pids)[tgt_of[target]["column"]]).to_numpy(
            dtype=float
        )
        X = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        # Keep anyone with finite y; per-feature NaNs are handled inside select/predict.
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
        if len(y) < MIN_N:
            skipped.append({**base, "reason": "n_lt_min", "n": int(len(y))})
            continue
        jobs.append(
            {
                **base,
                "k": 1,
                "cols_pool": cols,
                "X": X,
                "y": y,
                "n": int(len(y)),
                "tau": float(thr[target][str(task)]),
            }
        )
    return jobs, skipped


def run_one(cell: dict[str, Any], n_perm: int, *, n_boot: int = N_BOOT) -> dict[str, Any]:
    seed = cell_seed(int(cell["task"]), cell["source"], cell["target"])
    X = cell["X"]
    y = cell["y"]
    cols = cell["cols_pool"]
    tau = float(cell["tau"])
    obs = lopo_signx(X, y, cols, tau, seed, n_boot=n_boot)
    top = obs["top"]
    top1 = top[0][0] if top else ""
    top1_n = top[0][1] if top else 0
    top2 = top[1][0] if len(top) > 1 else ""

    p = float("nan")
    null_med = float("nan")
    null_p95 = float("nan")
    if n_perm > 0 and np.isfinite(obs["rho"]):
        rng = np.random.default_rng(SEED_PERM + seed)
        null = np.empty(n_perm, dtype=float)
        for pi in range(n_perm):
            yp = rng.permutation(y)
            null[pi] = lopo_signx(
                X, yp, cols, tau, seed + 10_000 + pi * 3, n_boot=n_boot
            )["rho"]
        null = null[np.isfinite(null)]
        extreme = int(np.sum(null >= float(obs["rho"]) - 1e-15))
        p = float((1 + extreme) / (1 + null.size)) if null.size else float("nan")
        null_med = float(np.median(null)) if null.size else float("nan")
        null_p95 = float(np.quantile(null, 0.95)) if null.size else float("nan")

    return {
        "task": cell["task"],
        "task_ja": cell["task_ja"],
        "source": cell["source"],
        "target": cell["target"],
        "k": 1,
        "n": cell["n"],
        "n_exp1_hits": cell["n_exp1_hits"],
        "n_pool": len(cols),
        "n_splits": cell["n"],  # LOPO
        "n_repeats": 1,
        "n_folds": cell["n"],
        "cols": top1,
        "kinds": feature_kind(top1) if top1 else "",
        "tau": tau,
        "stage_frac": obs["stage_frac"],
        "npass_mean": obs["npass_mean"],
        "top1_frac": float(top1_n / cell["n"]) if cell["n"] else 0.0,
        "top2_feat": top2,
        "rho": obs["rho"],
        "rho_sd": float("nan"),
        "rho_pos_frac": float("nan"),
        "null_med": null_med,
        "null_p95": null_p95,
        "p": p,
        "method": "lopo_stageA_signx",
    }
