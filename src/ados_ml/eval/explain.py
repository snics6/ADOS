"""Attention summaries and explanation pass/fail checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from ados_ml.data.tasks import TARGET_TASK_IDS, TASK_NAMES


def mean_attention_by_target(
    attentions: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """attentions: list over batches/folds of {target: (B,T)} -> mean (T,) per target."""
    keys = attentions[0].keys()
    out = {}
    for k in keys:
        stacked = np.concatenate([a[k] for a in attentions], axis=0)
        out[k] = stacked.mean(axis=0)
    return out


def c2_pretend_attention_pass(
    alpha_c2: np.ndarray,
    *,
    pretend_index: int,
    task_ids: tuple[int, ...] = TARGET_TASK_IDS,
) -> dict[str, Any]:
    """alpha_c2: (T,) mean attention for C2."""
    pretend_mass = float(alpha_c2[pretend_index])
    others = [float(alpha_c2[i]) for i in range(len(alpha_c2)) if i != pretend_index]
    ok = pretend_mass >= max(others) if others else True
    detail = {
        TASK_NAMES[tid]: float(alpha_c2[i]) for i, tid in enumerate(task_ids)
    }
    return {
        "pass": bool(ok),
        "pretend_mass": pretend_mass,
        "max_other": float(max(others)) if others else float("nan"),
        "attention_by_task": detail,
    }


def c2_ablation_pass(
    mae_full: float, mae_no_pretend: float
) -> dict[str, Any]:
    delta = float(mae_no_pretend - mae_full)
    # FAIL if removing pretend *improves* C2 (negative delta beyond tiny noise)
    fail = delta < -1e-6
    return {
        "pass": (not fail),
        "mae_full": mae_full,
        "mae_no_pretend": mae_no_pretend,
        "delta_mae": delta,
        "note": "PASS if no-pretend does not improve MAE",
    }
