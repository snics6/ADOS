"""Shared Exp3/4 pair labels: Exp2 rank-1, FDR both, near/far from kinds.

Write labels before transplant ρ or pool ρ.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from ados_ffm.data import TASK_JA, apply_target_y, group_of
from ados_ffm.ridge_cv import MIN_N, SOURCES, cv_schedule, perm_p
from ados_ffm.kinds import feature_kind
from ados_ffm.metrics import bh_fdr

TARGETS_REG = ("SA", "RRB", "CSS")
N_LABEL_PERM = 1000
SEED_LABEL = 54001
SEED_SIGN = 54101
SEED_CV_DIR = 55001

LABEL_COLS = (
    "task_a",
    "task_a_ja",
    "task_b",
    "task_b_ja",
    "source",
    "target",
    "col_a",
    "col_b",
    "kind_a",
    "kind_b",
    "label",
    "n_exp2_a",
    "n_exp2_b",
)


def rank1_col(cols: str) -> str:
    parts = [x for x in str(cols).split("|") if x]
    if not parts:
        raise ValueError(f"empty cols: {cols!r}")
    return parts[0]


def load_exp2_fdr(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fdr_sig" not in df.columns:
        raise KeyError("exp2 cells need fdr_sig")
    out = df.loc[df["fdr_sig"]].copy()
    out["col_rank1"] = out["cols"].map(rank1_col)
    out["kind_rank1"] = out["col_rank1"].map(feature_kind)
    return out


def lock_labels(exp2: pd.DataFrame) -> pd.DataFrame:
    """Near/far from rank-1 kinds. No ρ. FDR cells only."""
    rows = []
    for (source, target), g in exp2.groupby(["source", "target"], dropna=False):
        cells = {int(r.task): r for r in g.itertuples(index=False)}
        for a, b in combinations(sorted(cells), 2):
            ca, cb = cells[a], cells[b]
            ka, kb = str(ca.kind_rank1), str(cb.kind_rank1)
            rows.append(
                {
                    "task_a": int(a),
                    "task_a_ja": TASK_JA[int(a)],
                    "task_b": int(b),
                    "task_b_ja": TASK_JA[int(b)],
                    "source": source,
                    "target": target,
                    "col_a": str(ca.col_rank1),
                    "col_b": str(cb.col_rank1),
                    "kind_a": ka,
                    "kind_b": kb,
                    "label": "近い" if ka == kb else "遠い",
                    "n_exp2_a": int(ca.n),
                    "n_exp2_b": int(cb.n),
                }
            )
    out = pd.DataFrame(rows, columns=list(LABEL_COLS))
    if out.empty:
        return out
    return out.sort_values(["source", "target", "task_a", "task_b"]).reset_index(drop=True)


def expand_sides(labels: pd.DataFrame) -> pd.DataFrame:
    """Each task pair becomes two sides: look from A, look from B."""
    rows = []
    for r in labels.itertuples(index=False):
        rows.append(
            {
                "task_onto": int(r.task_a),
                "task_onto_ja": r.task_a_ja,
                "task_from": int(r.task_b),
                "task_from_ja": r.task_b_ja,
                "source": r.source,
                "target": r.target,
                "col_own": r.col_a,
                "col_from": r.col_b,
                "kind_own": r.kind_a,
                "kind_from": r.kind_b,
                "label": r.label,
            }
        )
        rows.append(
            {
                "task_onto": int(r.task_b),
                "task_onto_ja": r.task_b_ja,
                "task_from": int(r.task_a),
                "task_from_ja": r.task_a_ja,
                "source": r.source,
                "target": r.target,
                "col_own": r.col_b,
                "col_from": r.col_a,
                "kind_own": r.kind_b,
                "kind_from": r.kind_a,
                "label": r.label,
            }
        )
    return pd.DataFrame(rows)


def people_frame(
    tab: pd.DataFrame,
    lab: pd.DataFrame,
    cols: list[str],
    target: dict[str, Any],
) -> pd.DataFrame | None:
    missing = [c for c in cols if c not in tab.columns]
    if missing:
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
    if fr.empty:
        return None
    return fr.reset_index(drop=True)


def people_ready(fr: pd.DataFrame) -> pd.DataFrame | None:
    if fr is None or len(fr) < MIN_N:
        return None
    if cv_schedule(int(len(fr))) is None:
        return None
    return fr.reset_index(drop=True)


def people_cols(
    tab: pd.DataFrame,
    lab: pd.DataFrame,
    cols: list[str],
    target: dict[str, Any],
) -> pd.DataFrame | None:
    return people_ready(people_frame(tab, lab, cols, target))


def directed_seed(task_onto: int, task_from: int, source: str, target: str) -> int:
    return (
        SEED_CV_DIR
        + int(task_onto) * 200
        + int(task_from) * 17
        + SOURCES.index(source) * 3
        + TARGETS_REG.index(target)
    )


def sign_flip_p(
    deltas: np.ndarray,
    n_perm: int,
    seed: int,
    side: str,
) -> tuple[float, float]:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan")
    obs = float(np.mean(d))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=d.size)
        null[i] = float(np.mean(d * signs))
    return obs, perm_p(obs, null, side=side)


def label_shuffle_p(
    near: np.ndarray,
    far: np.ndarray,
    n_perm: int,
    seed: int,
    side: str,
) -> tuple[float, float]:
    near = np.asarray(near, dtype=float)
    far = np.asarray(far, dtype=float)
    near = near[np.isfinite(near)]
    far = far[np.isfinite(far)]
    if near.size == 0 or far.size == 0:
        return float("nan"), float("nan")
    obs = float(np.mean(far) - np.mean(near))
    labeled = np.concatenate([near, far])
    n_near = near.size
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(labeled)
        null[i] = float(np.mean(perm[n_near:]) - np.mean(perm[:n_near]))
    return obs, perm_p(obs, null, side=side)


def attach_bh(out: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    finite = out[p_col].notna() & np.isfinite(out[p_col].to_numpy(dtype=float))
    q = np.full(len(out), np.nan)
    m = int(finite.sum())
    if m:
        q[finite.to_numpy()] = bh_fdr(out.loc[finite, p_col].to_numpy(dtype=float))
    out = out.copy()
    out["q_fdr"] = q
    out["m"] = m
    out["fdr_sig"] = out["q_fdr"] < 0.05
    out["planned_m"] = 9
    return out


def planned_combos() -> list[tuple[str, str]]:
    return [(s, t) for s in SOURCES for t in TARGETS_REG]


def combo_seed(base: int, source: str, target: str) -> int:
    return base + SOURCES.index(source) * 3 + TARGETS_REG.index(target)


def main_from_sides(
    sides: pd.DataFrame,
    delta_col: str,
    *,
    n_perm: int,
    test_side: str,
) -> pd.DataFrame:
    """One test per source×score among far sides. Missing far → skip, not in m."""
    rows = []
    for source, target in planned_combos():
        g = sides[(sides["source"] == source) & (sides["target"] == target)]
        far = g.loc[g["label"] == "遠い", delta_col].to_numpy(dtype=float)
        near = g.loc[g["label"] == "近い", delta_col].to_numpy(dtype=float)
        far = far[np.isfinite(far)]
        near = near[np.isfinite(near)]
        rec = {
            "source": source,
            "target": target,
            "n_far": int(far.size),
            "n_near": int(near.size),
            "mean_delta_far": float(np.mean(far)) if far.size else float("nan"),
            "mean_delta_near": float(np.mean(near)) if near.size else float("nan"),
            "p": float("nan"),
            "skip": "",
        }
        if far.size == 0:
            rec["skip"] = "no_far"
            rows.append(rec)
            continue
        obs, p = sign_flip_p(
            far, n_perm, combo_seed(SEED_SIGN, source, target), test_side
        )
        rec["mean_delta_far"] = obs
        rec["p"] = p
        rows.append(rec)
    return attach_bh(pd.DataFrame(rows))


def secondary_from_sides(
    sides: pd.DataFrame,
    delta_col: str,
    *,
    n_perm: int,
    contrast: str,
) -> pd.DataFrame:
    """Need both near and far. contrast: far_minus_near | near_minus_far."""
    rows = []
    for source, target in planned_combos():
        g = sides[(sides["source"] == source) & (sides["target"] == target)]
        far = g.loc[g["label"] == "遠い", delta_col].to_numpy(dtype=float)
        near = g.loc[g["label"] == "近い", delta_col].to_numpy(dtype=float)
        far = far[np.isfinite(far)]
        near = near[np.isfinite(near)]
        rec = {
            "source": source,
            "target": target,
            "n_far": int(far.size),
            "n_near": int(near.size),
            "obs": float("nan"),
            "p": float("nan"),
            "skip": "",
        }
        if near.size == 0 or far.size == 0:
            rec["skip"] = "need_near_and_far"
            rows.append(rec)
            continue
        seed = combo_seed(SEED_LABEL, source, target)
        if contrast == "far_minus_near":
            obs, p = label_shuffle_p(near, far, n_perm, seed, "greater")
        elif contrast == "near_minus_far":
            obs, p = label_shuffle_p(far, near, n_perm, seed, "greater")
        else:
            raise ValueError(contrast)
        rec["obs"] = obs
        rec["p"] = p
        rows.append(rec)
    return attach_bh(pd.DataFrame(rows))
