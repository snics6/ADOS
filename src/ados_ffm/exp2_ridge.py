"""Experiment 2 auxiliary: Ridge on Exp1 FDR hits as-is. k=1 if 1 hit, else top 2.

Not the mainline. Mainline is LOPO sign(ρ)·x (`ados_ffm.exp2`).
BH family is (task × score). Writes `outputs/exp2_ridge/`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ados_ffm.data import TASK_JA, apply_target_y, group_of
from ados_ffm.exp1_univariate import merge_task_catalog
from ados_ffm.ridge_cv import (
    MIN_N,
    N_PERM,
    SEED_CV,
    SEED_PERM,
    SOURCES,
    cv_schedule,
    perm_cell_ridge,
    run_ridge_cols_cv,
)
from ados_ffm.kinds import feature_kind
from ados_ffm.metrics import bh_fdr

HIT_PATH_DEFAULT = "outputs/exp1/hits_main.csv"


def rank_hits(hits: pd.DataFrame, task: int, source: str, target: str) -> pd.DataFrame:
    sub = hits[
        (hits["task"] == int(task))
        & (hits["source"] == source)
        & (hits["target"] == target)
    ].copy()
    if sub.empty:
        return sub
    return sub.sort_values(
        ["q_fdr", "rho", "stability", "feature"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def k_from_hits(n_hits: int) -> int | None:
    if n_hits <= 0:
        return None
    if n_hits == 1:
        return 1
    return 2


def frame_for_cols(
    tab: pd.DataFrame,
    lab: pd.DataFrame,
    cols: list[str],
    target: dict[str, Any],
) -> pd.DataFrame | None:
    if any(c not in tab.columns for c in cols):
        return None
    fr = tab[["participant_id", *cols]].copy()
    fr["participant_id"] = fr["participant_id"].astype(str)
    y = apply_target_y(lab.reindex(fr["participant_id"])[target["column"]])
    fr = fr.assign(y=y.to_numpy(), group=[group_of(p, lab) for p in fr["participant_id"]])
    x = fr[cols].apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(fr["y"].to_numpy(dtype=float)) & np.isfinite(
        x.to_numpy(dtype=float)
    ).all(axis=1)
    fr = fr.loc[ok].copy()
    n = int(len(fr))
    if n < MIN_N or cv_schedule(n) is None:
        return None
    return fr.reset_index(drop=True)


def cell_seed(task: int, source: str, target: str) -> int:
    return (
        SEED_CV
        + int(task) * 30
        + SOURCES.index(source) * 3
        + (0 if target == "SA" else 1 if target == "RRB" else 2)
    )


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


def prepare_jobs(
    hits: pd.DataFrame,
    dyn: pd.DataFrame,
    new: pd.DataFrame,
    lab: pd.DataFrame,
    cohort: list[str],
    targets: tuple[dict[str, Any], ...],
    task_ids: tuple[int, ...],
    sources: tuple[str, ...] = SOURCES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tgt_of = {t["name"]: t for t in targets}
    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for task in task_ids:
        tab = merge_task_catalog(dyn, new, int(task), cohort)
        for source in sources:
            for tname in tgt_of:
                ranked = rank_hits(hits, int(task), source, tname)
                n_hits = int(len(ranked))
                k = k_from_hits(n_hits)
                base = {
                    "task": int(task),
                    "task_ja": TASK_JA[int(task)],
                    "source": source,
                    "target": tname,
                    "n_exp1_hits": n_hits,
                }
                if k is None:
                    skipped.append({**base, "reason": "no_exp1_hit"})
                    continue
                cols = [str(x) for x in ranked["feature"].tolist()[:k]]
                fr = frame_for_cols(tab, lab, cols, tgt_of[tname])
                if fr is None:
                    skipped.append(
                        {**base, "reason": "n_lt_12_or_missing", "cols": "|".join(cols), "k": k}
                    )
                    continue
                ns, nr = cv_schedule(int(len(fr)))
                kinds = [feature_kind(c) for c in cols]
                jobs.append(
                    {
                        **base,
                        "k": k,
                        "cols": cols,
                        "kinds": kinds,
                        "frame": fr,
                        "n": int(len(fr)),
                        "n_splits": ns,
                        "n_repeats": nr,
                        "exp1_rho": [float(x) for x in ranked["rho"].tolist()[:k]],
                        "exp1_q": [float(x) for x in ranked["q_fdr"].tolist()[:k]],
                        "exp1_stab": [float(x) for x in ranked["stability"].tolist()[:k]],
                    }
                )
    return jobs, skipped


def run_one(cell: dict[str, Any], n_perm: int) -> dict[str, Any]:
    seed = cell_seed(int(cell["task"]), cell["source"], cell["target"])
    fit = run_ridge_cols_cv(
        cell["frame"], cell["cols"], cell["n_splits"], cell["n_repeats"], seed
    )
    pval = perm_cell_ridge(
        cell["frame"],
        cell["cols"],
        cell["n_splits"],
        cell["n_repeats"],
        n_perm,
        SEED_PERM + seed,
        float(fit["rho"]),
    )
    return {
        "task": cell["task"],
        "task_ja": cell["task_ja"],
        "source": cell["source"],
        "target": cell["target"],
        "k": cell["k"],
        "n": cell["n"],
        "n_exp1_hits": cell["n_exp1_hits"],
        "n_splits": cell["n_splits"],
        "n_repeats": cell["n_repeats"],
        "n_folds": fit["n_folds"],
        "cols": "|".join(cell["cols"]),
        "kinds": "|".join(cell["kinds"]),
        "exp1_rho": "|".join(f"{x:.6g}" for x in cell["exp1_rho"]),
        "exp1_q": "|".join(f"{x:.6g}" for x in cell["exp1_q"]),
        "exp1_stab": "|".join(f"{x:.6g}" for x in cell["exp1_stab"]),
        "rho": fit["rho"],
        "rho_sd": fit["rho_sd"],
        "rho_pos_frac": fit["rho_pos_frac"],
        "p": pval,
    }
