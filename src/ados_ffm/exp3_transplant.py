"""Experiment 3: own rank-1 vs other task's rank-1, both on this task's video."""

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
    people_cols,
    secondary_from_sides,
)


def run_side(
    row: dict[str, Any],
    tabs: dict[int, pd.DataFrame],
    lab: pd.DataFrame,
    target: dict[str, Any],
) -> dict[str, Any] | None:
    onto = int(row["task_onto"])
    own = str(row["col_own"])
    other = str(row["col_from"])
    tab = tabs.get(onto)
    if tab is None or tab.empty:
        return None
    fr = people_cols(tab, lab, [own, other], target)
    if fr is None:
        return None
    n = int(len(fr))
    n_splits, n_repeats = cv_schedule(n)
    seed = directed_seed(onto, int(row["task_from"]), str(row["source"]), str(row["target"]))
    rho_own = run_ridge_cols_cv(fr, [own], n_splits, n_repeats, seed)["rho"]
    rho_from = run_ridge_cols_cv(fr, [other], n_splits, n_repeats, seed)["rho"]
    return {
        **row,
        "n": n,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "rho_own": float(rho_own),
        "rho_from": float(rho_from),
        "delta3": float(rho_own) - float(rho_from),
    }


def tests(sides: pd.DataFrame, n_perm: int = N_LABEL_PERM) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = main_from_sides(sides, "delta3", n_perm=n_perm, test_side="greater")
    sec = secondary_from_sides(
        sides, "delta3", n_perm=n_perm, contrast="far_minus_near"
    )
    return main, sec


__all__ = [
    "expand_sides",
    "load_exp2_fdr",
    "lock_labels",
    "run_side",
    "tests",
]
