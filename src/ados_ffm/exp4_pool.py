"""Experiment 4: average own rank-1 across two tasks vs this task alone."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ados_ffm.ridge_cv import cv_schedule, run_ridge_cols_cv
from ados_ffm.exp34_labels import (
    N_LABEL_PERM,
    directed_seed,
    expand_sides,
    load_exp2_fdr,
    lock_labels,
    main_from_sides,
    people_frame,
    people_ready,
    secondary_from_sides,
)


def run_side(
    row: dict[str, Any],
    tabs: dict[int, pd.DataFrame],
    lab: pd.DataFrame,
    target: dict[str, Any],
) -> dict[str, Any] | None:
    onto = int(row["task_onto"])
    other = int(row["task_from"])
    col = str(row["col_own"])
    ta, tb = tabs.get(onto), tabs.get(other)
    if ta is None or tb is None or ta.empty or tb.empty:
        return None
    pa = people_frame(ta, lab, [col], target)
    pb = people_frame(tb, lab, [col], target)
    if pa is None or pb is None:
        return None
    both = sorted(set(pa["participant_id"]) & set(pb["participant_id"]))
    pa = pa[pa["participant_id"].isin(both)].sort_values("participant_id").reset_index(drop=True)
    pb = pb[pb["participant_id"].isin(both)].sort_values("participant_id").reset_index(drop=True)
    fr = people_ready(pa)
    if fr is None:
        return None
    xa = pa[col].to_numpy(dtype=float)
    xb = pb[col].to_numpy(dtype=float)
    mix = fr.copy()
    mix[col] = 0.5 * (xa + xb)
    n = int(len(fr))
    n_splits, n_repeats = cv_schedule(n)
    seed = directed_seed(onto, other, str(row["source"]), str(row["target"]))
    rho_solo = run_ridge_cols_cv(fr, [col], n_splits, n_repeats, seed)["rho"]
    rho_mix = run_ridge_cols_cv(mix, [col], n_splits, n_repeats, seed)["rho"]
    return {
        **row,
        "n": n,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "rho_solo": float(rho_solo),
        "rho_mix": float(rho_mix),
        "delta4": float(rho_mix) - float(rho_solo),
    }


def tests(sides: pd.DataFrame, n_perm: int = N_LABEL_PERM) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = main_from_sides(sides, "delta4", n_perm=n_perm, test_side="less")
    sec = secondary_from_sides(
        sides, "delta4", n_perm=n_perm, contrast="near_minus_far"
    )
    return main, sec


__all__ = [
    "expand_sides",
    "load_exp2_fdr",
    "lock_labels",
    "run_side",
    "tests",
]
