"""Experiment 1: one-feature half-split stability screen + permutation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ados_ffm.data import FEATURE_COLS, SOURCE_OF, apply_target_y
from ados_extract.windows import feature_family, is_asr_risky
from ados_ffm.metrics import bh_fdr

EXTRA_PREFIXES: tuple[str, ...] = ("win_", "rsp_", "ges_", "txt_")
N_BOOT = 200
N_NULL = 50
THRESH_Q = 0.95
N_PERM = 1000
MIN_N = 8
SEED_NULL = 42001
SEED_REAL = 41001
SEED_PERM = 43001


def stable_int(text: str) -> int:
    h = 0
    for ch in text:
        h = (h * 131 + ord(ch)) % 1_000_003
    return h


def catalog_source(name: str) -> str:
    if name in SOURCE_OF:
        return SOURCE_OF[name]
    if name.startswith(("win_child_", "ges_child_", "txt_child_")):
        return "child"
    if name.startswith(("win_examiner_", "ges_examiner_", "txt_examiner_")):
        return "examiner"
    if name.startswith("rsp_"):
        return "dyad"
    return "other"


def is_catalog_col(name: str) -> bool:
    if name in FEATURE_COLS:
        return True
    return name.startswith(EXTRA_PREFIXES)


def catalog_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if is_catalog_col(c)]


def spearman_rho_fast(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3:
        return float("nan")
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    sx = float(np.dot(rx, rx))
    sy = float(np.dot(ry, ry))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / np.sqrt(sx * sy))


RHO_MIN_HALF = 0.20


def half_stability(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_rep: int = N_BOOT,
    rho_min: float = RHO_MIN_HALF,
    rng: np.random.Generator,
    positive_only: bool = True,
) -> float:
    """Non-overlapping halves. Default: both ρ >= rho_min (positive only)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)
    k = n // 2
    if k < 5:
        return float("nan")
    hits = 0
    for _ in range(n_rep):
        idx = rng.permutation(n)
        r1 = spearman_rho_fast(x[idx[:k]], y[idx[:k]])
        r2 = spearman_rho_fast(x[idx[k : k + k]], y[idx[k : k + k]])
        if not (np.isfinite(r1) and np.isfinite(r2)):
            continue
        if positive_only:
            ok = r1 >= rho_min and r2 >= rho_min
        else:
            ok = r1 * r2 > 0.0 and abs(r1) >= rho_min and abs(r2) >= rho_min
        if ok:
            hits += 1
    return float(hits / n_rep)


def two_sided_perm_p(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return float("nan")
    extreme = int(np.sum(np.abs(null) >= abs(float(observed))))
    return float((1 + extreme) / (1 + null.size))


def merge_task_catalog(
    dyn: pd.DataFrame,
    new: pd.DataFrame,
    task_id: int,
    cohort_ids: list[str],
) -> pd.DataFrame:
    """One row per participant: frozen 34 + shared extra columns."""
    ids = set(str(x) for x in cohort_ids)
    d = dyn[(dyn["task_id"] == task_id) & dyn["participant_id"].astype(str).isin(ids)].copy()
    d["participant_id"] = d["participant_id"].astype(str)
    n = pd.DataFrame()
    if new is not None and not new.empty:
        n = new[(new["task_id"] == task_id) & new["participant_id"].astype(str).isin(ids)].copy()
        n["participant_id"] = n["participant_id"].astype(str)
    extra = [c for c in (n.columns if not n.empty else []) if is_catalog_col(c)]
    keep_d = ["participant_id"] + [c for c in FEATURE_COLS if c in d.columns]
    if d.empty and (n.empty or not extra):
        return pd.DataFrame()
    if d.empty:
        out = n[["participant_id"] + extra].copy()
        for c in FEATURE_COLS:
            if c not in out.columns:
                out[c] = np.nan
        return out
    out = d[keep_d]
    if extra:
        out = out.merge(n[["participant_id"] + extra], on="participant_id", how="outer")
    return out.reset_index(drop=True)


def feature_payloads(
    frame: pd.DataFrame,
    y_by_pid: dict[str, float],
    cols: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for col in cols:
        if col not in frame.columns:
            continue
        xraw = pd.to_numeric(frame[col], errors="coerce")
        pids, xs, ys = [], [], []
        for pid, xv in zip(frame["participant_id"].astype(str), xraw):
            yv = y_by_pid.get(pid)
            if yv is None or not np.isfinite(yv) or not np.isfinite(float(xv)):
                continue
            pids.append(pid)
            xs.append(float(xv))
            ys.append(float(yv))
        if len(pids) < MIN_N:
            continue
        x = np.asarray(xs, dtype=float)
        if float(np.std(x)) <= 0.0:
            continue
        out.append(
            {
                "feature": col,
                "source": catalog_source(col),
                "family": "frozen34" if col in FEATURE_COLS else feature_family(col),
                "asr_risky": bool(is_asr_risky(col)),
                "pids": pids,
                "x": x,
                "y": np.asarray(ys, dtype=float),
                "n": len(pids),
            }
        )
    return out


def y_map_for_task(
    frame: pd.DataFrame,
    lab: pd.DataFrame,
    target: dict[str, Any],
) -> dict[str, float]:
    pids = frame["participant_id"].astype(str)
    raw = lab.reindex(pids)[target["column"]]
    y = apply_target_y(raw)
    out: dict[str, float] = {}
    for pid, yv in zip(pids, y.to_numpy()):
        try:
            fv = float(yv)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out[str(pid)] = fv
    return out


def null_stabilities_one_target(
    payloads: list[dict[str, Any]],
    pids_task: list[str],
    y_by_pid: dict[str, float],
    *,
    n_null: int,
    n_boot: int,
    seed: int,
    positive_only: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y0 = np.asarray([y_by_pid[p] for p in pids_task], dtype=float)
    for si in range(n_null):
        rng_s = np.random.default_rng(seed + si)
        y_shuf = y0.copy()
        rng_s.shuffle(y_shuf)
        ymap = {p: float(v) for p, v in zip(pids_task, y_shuf)}
        for fi, pl in enumerate(payloads):
            y = np.asarray([ymap[p] for p in pl["pids"]], dtype=float)
            rng_b = np.random.default_rng(seed + 10_000 + si * 1_000 + fi)
            st = half_stability(
                pl["x"], y, n_rep=n_boot, rng=rng_b, positive_only=positive_only
            )
            rows.append(
                {
                    "shuffle": si,
                    "feature": pl["feature"],
                    "stability": st,
                    "n": pl["n"],
                }
            )
    return rows


def real_stabilities(
    payloads: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
    positive_only: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fi, pl in enumerate(payloads):
        rng = np.random.default_rng(seed + fi)
        st = half_stability(
            pl["x"], pl["y"], n_rep=n_boot, rng=rng, positive_only=positive_only
        )
        rho = spearman_rho_fast(pl["x"], pl["y"])
        rows.append(
            {
                "feature": pl["feature"],
                "source": pl["source"],
                "family": pl["family"],
                "asr_risky": pl["asr_risky"],
                "n": pl["n"],
                "stability": st,
                "rho": rho,
            }
        )
    return rows


def perm_one(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int,
    seed: int,
    two_sided: bool = True,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    obs = spearman_rho_fast(x, y)
    null = np.empty(n_perm, dtype=float)
    yw = y.copy()
    for i in range(n_perm):
        rng.shuffle(yw)
        null[i] = spearman_rho_fast(x, yw)
    if not np.isfinite(obs):
        p = float("nan")
    elif two_sided:
        p = two_sided_perm_p(obs, null)
    elif obs <= 0.0:
        p = float("nan")
    else:
        extreme = int(np.sum(null >= obs))
        p = float((1 + extreme) / (1 + n_perm))
    return {
        "rho": float(obs) if np.isfinite(obs) else float("nan"),
        "p": p,
        "null_med": float(np.nanmedian(null)),
        "null_p95": float(np.nanpercentile(null, 95)),
    }


def attach_fdr(df: pd.DataFrame, by: tuple[str, ...] = ("task", "target")) -> pd.DataFrame:
    """BH within each (task, target) family. Writes q, m, fdr_sig."""
    out = df.copy()
    out["q_fdr"] = np.nan
    out["q_family"] = ""
    out["fdr_sig"] = False
    out["m"] = 0
    if out.empty:
        return out
    for keys, g in out.groupby(list(by), dropna=False):
        p = g["p"].to_numpy(dtype=float)
        q = bh_fdr(p)
        m = int(np.isfinite(p).sum())
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = "_".join(str(k) for k in keys)
        out.loc[g.index, "q_fdr"] = q
        out.loc[g.index, "q_family"] = label
        out.loc[g.index, "fdr_sig"] = np.asarray(q) < 0.05
        out.loc[g.index, "m"] = m
    return out
