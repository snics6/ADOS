"""Non-tautological Exp3 findings: r, (λ,r) positions, duration, Q ranks, initiator.

The pooling identity and the λ* curve are algebra. This module records where
the data fell, whether duration explains the signs, where flip families sit
in the unselected Q ranking, and whether chain sign-flips split by initiator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ados_extract.dynamics.conversation import chain_initiator_features
from ados_extract.task_segments import load_task_segments_map, merge_spans
from ados_ffm.data import ROOT, TASK_JA, TASKS, load_cohort, load_labels, apply_target_y
from ados_ffm.exp1_univariate import spearman_rho_fast
from ados_ffm.hetero import lambda_star

DATA_ROOT = ROOT / "data"
FLIP_KEYS = [
    ("dyad_chain_ge4_per_min", "CSS"),
    ("dyad_chain_ge4_per_min", "SA"),
    ("dyad_chain_max", "SA"),
    ("dyad_chain_mean", "SA"),
    ("examiner_turn_rate_per_min", "SA"),
    ("win_child_speech_frac_delta", "CSS"),
    ("win_child_speech_frac_delta", "SA"),
    ("win_child_turn_rate_delta", "SA"),
]
PHENOMENON = {
    ("dyad_chain_ge4_per_min", "CSS"): "ja_free",
    ("dyad_chain_ge4_per_min", "SA"): "ja_free",
    ("dyad_chain_max", "SA"): "ja_free",
    ("dyad_chain_mean", "SA"): "ja_free",
    ("examiner_turn_rate_per_min", "SA"): "ja_free",
    ("win_child_speech_frac_delta", "CSS"): "speech_drift",
    ("win_child_speech_frac_delta", "SA"): "speech_drift",
    ("win_child_turn_rate_delta", "SA"): "speech_drift",
}


def spearman_partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman ρ(x, y | z) via the rank-Pearson partial formula."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(m.sum()) < 8:
        return float("nan")
    rx = rankdata(x[m], method="average")
    ry = rankdata(y[m], method="average")
    rz = rankdata(z[m], method="average")
    rxy = float(np.corrcoef(rx, ry)[0, 1])
    rxz = float(np.corrcoef(rx, rz)[0, 1])
    ryz = float(np.corrcoef(ry, rz)[0, 1])
    den = np.sqrt(max(0.0, 1.0 - rxz**2) * max(0.0, 1.0 - ryz**2))
    if den <= 1e-12:
        return float("nan")
    return float((rxy - rxz * ryz) / den)


def q_ranks(families: pd.DataFrame) -> pd.DataFrame:
    fam = families.copy()
    fam["p_rank"] = fam["p"].rank(method="min")
    fam["n_fam"] = int(np.isfinite(fam["p"]).sum())
    rows = []
    for feat, tgt in FLIP_KEYS + [("ges_child_fidgeting_per_min", "CSS")]:
        hit = fam[(fam["feature"] == feat) & (fam["target"] == tgt)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append(
            {
                "feature": feat,
                "target": tgt,
                "phenomenon": PHENOMENON.get((feat, tgt), "other"),
                "p": float(r["p"]),
                "q_fdr": float(r["q_fdr"]),
                "p_rank": int(r["p_rank"]),
                "n_fam": int(r["n_fam"]),
                "Q": float(r["Q"]),
                "I2": float(r["I2"]),
                "n_sign_discord": int(r["n_sign_discord"]),
            }
        )
    return pd.DataFrame(rows)


def duration_by_task(dyn: pd.DataFrame, cohort: list[str]) -> pd.DataFrame:
    d = dyn[dyn["participant_id"].astype(str).isin(cohort)].copy()
    rows = []
    for tid, g in d.groupby("task_id"):
        v = pd.to_numeric(g["duration_sec"], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        rows.append(
            {
                "task": int(tid),
                "task_ja": TASK_JA[int(tid)],
                "n": int(v.size),
                "mean_sec": float(np.mean(v)),
                "sd_sec": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
                "min_sec": float(np.min(v)),
                "q25_sec": float(np.quantile(v, 0.25)),
                "median_sec": float(np.median(v)),
                "q75_sec": float(np.quantile(v, 0.75)),
                "max_sec": float(np.max(v)),
            }
        )
    return pd.DataFrame(rows).sort_values("task")


SEED_INIT = 91001
N_BOOT_INIT = 2000
RHO_MIN_UNSEL = 0.20

TASK_EN = {
    1: "1 construct",
    2: "2 pretend",
    3: "3 JA",
    4: "4 demo",
    5: "5 picture",
    6: "6 book",
    7: "7 free play",
    8: "8 birthday",
    9: "9 snack",
    10: "10 routine",
}


def duration_wide(dyn: pd.DataFrame, cohort: list[str]) -> pd.DataFrame:
    d = dyn[dyn["participant_id"].astype(str).isin(cohort)].copy()
    d["participant_id"] = d["participant_id"].astype(str)
    d["task_id"] = d["task_id"].astype(int)
    d["duration_sec"] = pd.to_numeric(d["duration_sec"], errors="coerce")
    return d.pivot_table(
        index="participant_id",
        columns="task_id",
        values="duration_sec",
        aggfunc="first",
    )


def duration_y_correlations(
    dyn: pd.DataFrame, lab: pd.DataFrame, cohort: list[str]
) -> pd.DataFrame:
    """Spearman Corr(duration_k, y) for every task × SA/RRB/CSS. No feature involved."""
    from ados_ffm.data import TARGETS

    wide = duration_wide(dyn, cohort)
    y_by = {
        tgt["name"]: apply_target_y(lab.reindex([str(p) for p in cohort])[tgt["column"]])
        for tgt in TARGETS
    }
    for y in y_by.values():
        y.index = [str(i) for i in y.index]
    rows = []
    for tid in TASKS:
        if tid not in wide.columns:
            continue
        x = pd.to_numeric(wide[tid], errors="coerce")
        for tgt, y in y_by.items():
            idx = sorted(set(x.index) & set(y.index))
            xv = x.reindex(idx).to_numpy(dtype=float)
            yv = y.reindex(idx).to_numpy(dtype=float)
            m = np.isfinite(xv) & np.isfinite(yv)
            rows.append(
                {
                    "task": int(tid),
                    "task_ja": TASK_JA[int(tid)],
                    "task_en": TASK_EN[int(tid)],
                    "target": tgt,
                    "n": int(m.sum()),
                    "rho": spearman_rho_fast(xv[m], yv[m]) if int(m.sum()) >= 8 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def pair_min_duration(wide: pd.DataFrame) -> pd.DataFrame:
    """Median of min(dur_A, dur_B) on the participant intersection, every task pair."""
    tasks = [int(c) for c in wide.columns]
    rows = []
    for i, ta in enumerate(tasks):
        for tb in tasks[i + 1 :]:
            a = pd.to_numeric(wide[ta], errors="coerce")
            b = pd.to_numeric(wide[tb], errors="coerce")
            m = np.isfinite(a) & np.isfinite(b)
            mind = np.minimum(a[m].to_numpy(dtype=float), b[m].to_numpy(dtype=float))
            rows.append(
                {
                    "task_a": int(ta),
                    "task_b": int(tb),
                    "n_overlap": int(m.sum()),
                    "min_dur_median": float(np.median(mind)) if mind.size else float("nan"),
                    "min_dur_q25": float(np.quantile(mind, 0.25)) if mind.size else float("nan"),
                    "min_dur_q75": float(np.quantile(mind, 0.75)) if mind.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def r_vs_duration_table(r_all: pd.DataFrame, pair_dur: pd.DataFrame) -> pd.DataFrame:
    out = r_all.merge(pair_dur, on=["task_a", "task_b"], how="left")
    return out


def r_vs_duration_summary(tab: pd.DataFrame) -> dict[str, Any]:
    ok = tab[np.isfinite(tab["r"]) & np.isfinite(tab["min_dur_median"])].copy()
    if ok.empty:
        return {}
    rho = spearman_rho_fast(
        ok["min_dur_median"].to_numpy(dtype=float),
        ok["r"].to_numpy(dtype=float),
    )
    q = pd.qcut(ok["min_dur_median"], 4, duplicates="drop")
    bins = (
        ok.groupby(q, observed=True)["r"]
        .agg(n="size", median="median", q25=lambda s: float(np.quantile(s, 0.25)), q75=lambda s: float(np.quantile(s, 0.75)))
        .reset_index(drop=True)
    )
    # also group by unique duration (45 task pairs)
    by_pair = ok.groupby(["task_a", "task_b"], as_index=False).agg(
        min_dur_median=("min_dur_median", "first"),
        r_median=("r", "median"),
        n=("r", "size"),
    )
    return {
        "n": int(len(ok)),
        "spearman_r_vs_min_dur": float(rho),
        "bin_medians": [float(x) for x in bins["median"].to_numpy()],
        "bin_n": [int(x) for x in bins["n"].to_numpy()],
        "pair_level_spearman": float(
            spearman_rho_fast(
                by_pair["min_dur_median"].to_numpy(dtype=float),
                by_pair["r_median"].to_numpy(dtype=float),
            )
        )
        if len(by_pair) >= 8
        else float("nan"),
    }


def q_p_curve_summary(families: pd.DataFrame) -> dict[str, Any]:
    p = pd.to_numeric(families["p"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(p)
    n = int(ok.sum())
    thresholds = (0.05, 0.10, 0.20)
    ecdf = []
    for thr in thresholds:
        n_lt = int((p[ok] < thr).sum()) if n else 0
        expected = float(thr * n) if n else float("nan")
        ecdf.append(
            {
                "threshold": float(thr),
                "n_obs": n_lt,
                "n_expected": expected,
                "excess": float(n_lt - expected) if n else float("nan"),
            }
        )
    n_lt05 = int(ecdf[0]["n_obs"]) if ecdf else 0
    return {
        "n_families": n,
        "n_p_lt_05": n_lt05,
        "expected_p_lt_05": float(0.05 * n) if n else float("nan"),
        "frac_p_lt_05": float(n_lt05 / n) if n else float("nan"),
        "ecdf": ecdf,
        "dependence_note": "features are correlated; describe the ECDF, do not test it",
    }


def unselected_lambda_cells(panel, r_all: pd.DataFrame, *, rho_min: float = RHO_MIN_UNSEL, min_n: int = 12) -> pd.DataFrame:
    """All feature × target × task-pair cells with |ρ_A|≥rho_min, oriented |ρ_A|≥|ρ_B|."""
    r_lookup: dict[tuple[str, int, int], float] = {}
    for rec in r_all.itertuples(index=False):
        r_lookup[(str(rec.feature), int(rec.task_a), int(rec.task_b))] = float(rec.r)
    rows = []
    for feat in panel.features:
        xmap = panel.X[feat]
        for tgt, y in panel.y.items():
            rho_t: dict[int, float] = {}
            for tid in panel.tasks:
                x = xmap[int(tid)]
                m = np.isfinite(x) & np.isfinite(y)
                n = int(m.sum())
                rho_t[int(tid)] = spearman_rho_fast(x[m], y[m]) if n >= min_n else float("nan")
            tasks = [int(t) for t in panel.tasks]
            for i, ta in enumerate(tasks):
                for tb in tasks[i + 1 :]:
                    ra, rb = rho_t[ta], rho_t[tb]
                    r = r_lookup.get((feat, ta, tb), float("nan"))
                    if not (np.isfinite(ra) and np.isfinite(rb) and np.isfinite(r)):
                        continue
                    if abs(rb) > abs(ra):
                        ra, rb, ta, tb = rb, ra, tb, ta
                    if abs(ra) < rho_min:
                        continue
                    lam = float(rb / ra) if ra != 0.0 else float("nan")
                    star = float(lambda_star(r))
                    rows.append(
                        {
                            "feature": feat,
                            "target": tgt,
                            "task_a": int(ta),
                            "task_b": int(tb),
                            "rho_a": float(ra),
                            "rho_b": float(rb),
                            "r": float(r),
                            "lambda": lam,
                            "lambda_star": star,
                            "pool_worse": bool(np.isfinite(lam) and lam < star),
                            "sign_flip": bool(ra * rb < 0.0),
                        }
                    )
    return pd.DataFrame(rows)


def unselected_lambda_summary(cells: pd.DataFrame) -> dict[str, Any]:
    if cells.empty:
        return {"n_cells": 0}
    n = int(len(cells))
    n_worse = int(cells["pool_worse"].sum())
    same = cells[~cells["sign_flip"].astype(bool)]
    return {
        "n_cells": n,
        "n_pool_worse": n_worse,
        "frac_pool_worse": float(n_worse / n),
        "n_sign_flip": int(cells["sign_flip"].sum()),
        "n_same_sign": int(len(same)),
        "frac_pool_worse_same_sign": float(same["pool_worse"].mean()) if len(same) else float("nan"),
        "rho_min": RHO_MIN_UNSEL,
    }


SEED_LAM_PERM = 92001
N_PERM_LAM = 500


def _stack_panel_X(panel) -> np.ndarray:
    feats = list(panel.features)
    tasks = [int(t) for t in panel.tasks]
    n = int(panel.n)
    X = np.full((len(feats), len(tasks), n), np.nan)
    for fi, feat in enumerate(feats):
        xmap = panel.X[feat]
        for ti, tid in enumerate(tasks):
            X[fi, ti] = np.asarray(xmap[int(tid)], dtype=float)
    return X


def _pair_star_grid(
    r_all: pd.DataFrame, features: list[str], tasks: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmap = {f: i for i, f in enumerate(features)}
    pairs = [(i, j) for i in range(len(tasks)) for j in range(i + 1, len(tasks))]
    pmap = {(tasks[a], tasks[b]): k for k, (a, b) in enumerate(pairs)}
    rmat = np.full((len(features), len(pairs)), np.nan)
    for rec in r_all.itertuples(index=False):
        fi = fmap.get(str(rec.feature))
        k = pmap.get((int(rec.task_a), int(rec.task_b)))
        if fi is None or k is None:
            continue
        rmat[fi, k] = float(rec.r)
    star = np.sqrt(np.maximum(0.0, 2.0 * (1.0 + rmat))) - 1.0
    ia = np.array([a for a, _b in pairs], dtype=int)
    ib = np.array([b for _a, b in pairs], dtype=int)
    return star, ia, ib


def _rho_grid(X: np.ndarray, y: np.ndarray, *, min_n: int = 12) -> np.ndarray:
    from ados_ffm.hetero import spearman_masked_batch

    F, T, n = X.shape
    y = np.asarray(y, dtype=float)
    yrow = np.broadcast_to(y, (F, n))
    out = np.full((F, T), np.nan)
    for t in range(T):
        rho, nn = spearman_masked_batch(X[:, t, :], yrow)
        out[:, t] = np.where(nn >= min_n, rho, np.nan)
    return out


def _lambda_frac(
    rho: np.ndarray, star: np.ndarray, ia: np.ndarray, ib: np.ndarray, *, rho_min: float
) -> dict[str, float]:
    ra = rho[:, ia]
    rb = rho[:, ib]
    swap = np.abs(rb) > np.abs(ra)
    ra2 = np.where(swap, rb, ra)
    rb2 = np.where(swap, ra, rb)
    ok = np.isfinite(ra2) & np.isfinite(rb2) & np.isfinite(star) & (np.abs(ra2) >= rho_min)
    lam = np.divide(rb2, ra2, out=np.full_like(ra2, np.nan), where=ra2 != 0.0)
    worse = lam < star
    n = int(ok.sum())
    same = ok & (ra2 * rb2 > 0.0)
    n_same = int(same.sum())
    return {
        "n_cells": float(n),
        "frac_pool_worse": float(worse[ok].mean()) if n else float("nan"),
        "n_same_sign": float(n_same),
        "frac_pool_worse_same_sign": float(worse[same].mean()) if n_same else float("nan"),
    }


def permute_lambda_frac(
    panel,
    r_all: pd.DataFrame,
    *,
    n_perm: int = N_PERM_LAM,
    seed: int = SEED_LAM_PERM,
    rho_min: float = RHO_MIN_UNSEL,
    min_n: int = 12,
) -> dict[str, Any]:
    """Subject-level y shuffle. X (and therefore r) stay aligned across tasks."""
    feats = list(panel.features)
    X = _stack_panel_X(panel)
    star, ia, ib = _pair_star_grid(r_all, feats, [int(t) for t in panel.tasks])
    y_list = [np.asarray(panel.y[tgt], dtype=float) for tgt in panel.y]
    n = int(panel.n)
    rng = np.random.default_rng(seed)
    fracs = np.full(n_perm, np.nan)
    fracs_same = np.full(n_perm, np.nan)
    n_cells = np.full(n_perm, np.nan)
    for b in range(n_perm):
        idx = rng.permutation(n)
        rhos = [_rho_grid(X, y[idx], min_n=min_n) for y in y_list]
        rho = np.concatenate(rhos, axis=0)
        star3 = np.concatenate([star] * len(rhos), axis=0)
        rec = _lambda_frac(rho, star3, ia, ib, rho_min=rho_min)
        fracs[b] = rec["frac_pool_worse"]
        fracs_same[b] = rec["frac_pool_worse_same_sign"]
        n_cells[b] = rec["n_cells"]
        if (b + 1) % 100 == 0 or b + 1 == n_perm:
            print(
                f"  λ-perm  {b + 1}/{n_perm}  null median={np.nanmedian(fracs[: b + 1]):.3f}",
                flush=True,
            )
    ok = np.isfinite(fracs)
    return {
        "n_perm": int(n_perm),
        "seed": int(seed),
        "null_median": float(np.median(fracs[ok])) if ok.any() else float("nan"),
        "null_q05": float(np.quantile(fracs[ok], 0.05)) if ok.any() else float("nan"),
        "null_q95": float(np.quantile(fracs[ok], 0.95)) if ok.any() else float("nan"),
        "null_mean": float(np.mean(fracs[ok])) if ok.any() else float("nan"),
        "null_same_median": float(np.nanmedian(fracs_same)) if ok.any() else float("nan"),
        "null_same_q95": float(np.nanquantile(fracs_same, 0.95)) if ok.any() else float("nan"),
        "null_n_cells_median": float(np.nanmedian(n_cells)) if ok.any() else float("nan"),
        "fracs": [float(x) for x in fracs],
        "fracs_same": [float(x) for x in fracs_same],
    }


def duration_partial_table(
    dyn: pd.DataFrame,
    win: pd.DataFrame,
    lab: pd.DataFrame,
    flips: pd.DataFrame,
    cohort: list[str],
) -> pd.DataFrame:
    """ρ(x, y) vs ρ(x, y | duration) on each side of a sign-flip combo."""
    dyn = dyn.copy()
    dyn["participant_id"] = dyn["participant_id"].astype(str)
    win = win.copy()
    win["participant_id"] = win["participant_id"].astype(str)
    from ados_ffm.data import TARGETS

    y_by = {
        tgt["name"]: apply_target_y(lab.reindex([str(p) for p in cohort])[tgt["column"]])
        for tgt in TARGETS
    }

    def _col(tab: pd.DataFrame, feat: str, tid: int) -> pd.Series:
        sub = tab[tab["task_id"].astype(int) == int(tid)]
        s = sub.set_index("participant_id")[feat]
        return pd.to_numeric(s, errors="coerce")

    rows = []
    for rec in flips.itertuples(index=False):
        feat = str(rec.feature)
        if feat.startswith("win_"):
            src = win
            dur_src = win
            dur_col = "new_duration_sec" if "new_duration_sec" in win.columns else "duration_sec"
        else:
            src = dyn
            dur_src = dyn
            dur_col = "duration_sec"
        y = y_by[str(rec.target)].copy()
        y.index = [str(i) for i in y.index]
        for side, tid in (("pos", int(rec.task_pos)), ("neg", int(rec.task_neg))):
            x = _col(src, feat, tid)
            d = _col(dur_src, dur_col, tid)
            idx = sorted(set(x.index) & set(d.index) & set(y.index))
            xv = x.reindex(idx).to_numpy(dtype=float)
            yv = y.reindex(idx).to_numpy(dtype=float)
            dv = d.reindex(idx).to_numpy(dtype=float)
            m = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(dv)
            rho = spearman_rho_fast(xv[m], yv[m])
            rho_d = spearman_rho_fast(xv[m], dv[m])
            rho_yd = spearman_rho_fast(yv[m], dv[m])
            rho_p = spearman_partial(xv, yv, dv)
            rows.append(
                {
                    "feature": feat,
                    "target": rec.target,
                    "side": side,
                    "task": int(tid),
                    "task_ja": TASK_JA[int(tid)],
                    "n": int(m.sum()),
                    "rho": float(rho),
                    "rho_x_duration": float(rho_d),
                    "rho_y_duration": float(rho_yd),
                    "rho_partial": float(rho_p),
                    "sign_survives": bool(
                        np.isfinite(rho)
                        and np.isfinite(rho_p)
                        and np.sign(rho) == np.sign(rho_p)
                        and abs(rho_p) > 1e-9
                    ),
                    "rate_normalized": "per_min" in feat,
                }
            )
    return pd.DataFrame(rows)


def extract_chain_initiators(
    cohort: list[str],
    *,
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    segs_map = load_task_segments_map(cohort)
    rows = []
    for i, pid in enumerate(cohort):
        path = data_root / pid / f"{pid}_multimodal_session_v1.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        speech = doc.get("speech_segments") or []
        # One row per (participant, task). A task that was interrupted and
        # resumed has several annotated stretches; they are its spans, and the
        # time between them is another activity, so they are neither separate
        # rows nor a single hull.
        spans_by_task: dict[int, list[tuple[float, float]]] = {}
        for s in segs_map.get(pid, []):
            tid = int(s.get("task_id", -1))
            if tid not in TASKS:
                continue
            spans_by_task.setdefault(tid, []).append(
                (float(s["session_start_sec"]), float(s["session_end_sec"]))
            )
        for tid in sorted(spans_by_task):
            spans = merge_spans(spans_by_task[tid])
            feat = chain_initiator_features(speech, spans)
            feat["participant_id"] = str(pid)
            feat["task_id"] = tid
            feat["duration_sec"] = float(sum(b - a for a, b in spans))
            feat["n_task_spans"] = float(len(spans))
            rows.append(feat)
        if (i + 1) % 10 == 0 or i + 1 == len(cohort):
            print(f"  chain-init  {i + 1}/{len(cohort)}", flush=True)
    return pd.DataFrame(rows)


def initiator_correlations(
    chains: pd.DataFrame,
    lab: pd.DataFrame,
    cohort: list[str],
    *,
    n_boot: int = N_BOOT_INIT,
    seed: int = SEED_INIT,
) -> pd.DataFrame:
    """JA (3) vs free play (7): overall / exam-init / child-init chains × SA and CSS."""
    from ados_ffm.data import TARGETS
    from ados_ffm.hetero import percentile_ci, spearman_complete_batch

    y_by = {
        tgt["name"]: apply_target_y(lab.reindex([str(p) for p in cohort])[tgt["column"]])
        for tgt in TARGETS
        if tgt["name"] in ("SA", "CSS")
    }
    cols = [
        "dyad_chain_ge4_per_min",
        "dyad_chain_ge4_per_min_exam_init",
        "dyad_chain_ge4_per_min_child_init",
        "dyad_chain_max",
        "dyad_chain_max_exam_init",
        "dyad_chain_max_child_init",
        "dyad_chain_mean",
        "dyad_chain_mean_exam_init",
        "dyad_chain_mean_child_init",
    ]
    rows = []
    cell_i = 0
    for tid in (3, 7):
        sub = chains[chains["task_id"].astype(int) == tid].set_index("participant_id")
        sub.index = sub.index.astype(str)
        for tgt, y in y_by.items():
            y = y.copy()
            y.index = [str(i) for i in y.index]
            for col in cols:
                if col not in sub.columns:
                    continue
                x = pd.to_numeric(sub[col], errors="coerce")
                idx = sorted(set(x.index) & set(y.index))
                xv = x.reindex(idx).to_numpy(dtype=float)
                yv = y.reindex(idx).to_numpy(dtype=float)
                d = pd.to_numeric(sub.reindex(idx)["duration_sec"], errors="coerce").to_numpy(dtype=float)
                m = np.isfinite(xv) & np.isfinite(yv)
                md = m & np.isfinite(d)
                xb, yb = xv[m], yv[m]
                n = int(xb.size)
                rho = spearman_rho_fast(xb, yb) if n >= 8 else float("nan")
                lo = hi = float("nan")
                if n >= 8 and n_boot >= 20:
                    rng = np.random.default_rng(seed + cell_i)
                    draw = rng.integers(0, n, size=(n_boot, n))
                    boots = spearman_complete_batch(xb[draw], yb[draw])
                    lo, hi = percentile_ci(boots)
                cell_i += 1
                rows.append(
                    {
                        "task": tid,
                        "task_ja": TASK_JA[tid],
                        "target": tgt,
                        "feature": col,
                        "who": (
                            "exam"
                            if "_exam_init" in col
                            else "child"
                            if "_child_init" in col
                            else "all"
                        ),
                        "n": n,
                        "rho": rho,
                        "rho_lo": lo,
                        "rho_hi": hi,
                        "rho_partial_duration": spearman_partial(xv, yv, d) if int(md.sum()) >= 8 else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def initiator_counts(chains: pd.DataFrame) -> pd.DataFrame:
    """Per-session chain and ≥4-chain counts, split by initiator. Independent of y."""
    rows = []
    for tid in (3, 7):
        sub = chains[chains["task_id"].astype(int) == tid].copy()
        if sub.empty:
            continue
        dur = pd.to_numeric(sub["duration_sec"], errors="coerce").to_numpy(dtype=float)
        specs = (
            ("all", "dyad_chain_n", "dyad_chain_ge4_per_min"),
            ("exam", "dyad_chain_n_exam_init", "dyad_chain_ge4_per_min_exam_init"),
            ("child", "dyad_chain_n_child_init", "dyad_chain_ge4_per_min_child_init"),
        )
        for who, n_col, ge4_col in specs:
            if n_col not in sub.columns or ge4_col not in sub.columns:
                continue
            n_ch = pd.to_numeric(sub[n_col], errors="coerce").to_numpy(dtype=float)
            ge4_pm = pd.to_numeric(sub[ge4_col], errors="coerce").to_numpy(dtype=float)
            n_ge4 = ge4_pm * (dur / 60.0)
            m_n = np.isfinite(n_ch)
            m_g = np.isfinite(n_ge4)
            rows.append(
                {
                    "task": int(tid),
                    "task_ja": TASK_JA[int(tid)],
                    "who": who,
                    "n_sessions": int(m_n.sum()),
                    "median_n_chains": float(np.median(n_ch[m_n])) if m_n.any() else float("nan"),
                    "mean_n_chains": float(np.mean(n_ch[m_n])) if m_n.any() else float("nan"),
                    "median_n_ge4": float(np.median(n_ge4[m_g])) if m_g.any() else float("nan"),
                    "mean_n_ge4": float(np.mean(n_ge4[m_g])) if m_g.any() else float("nan"),
                    "frac_zero_ge4": float(np.mean(n_ge4[m_g] <= 0.0)) if m_g.any() else float("nan"),
                    "median_duration_sec": float(np.median(dur[np.isfinite(dur)])) if np.isfinite(dur).any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def pairs21_as_flips(pairs: pd.DataFrame) -> pd.DataFrame:
    """The 8 sign-flip rows of pairs_21, oriented so task_pos has ρ>0."""
    rows = []
    for rec in pairs[pairs["sign_flip_cap"].astype(bool)].itertuples(index=False):
        ra = float(rec.rho_cap_a)
        if ra >= 0.0:
            tpos, tneg = int(rec.task_a), int(rec.task_b)
        else:
            tpos, tneg = int(rec.task_b), int(rec.task_a)
        rows.append(
            {
                "feature": rec.feature,
                "target": rec.target,
                "task_pos": tpos,
                "task_neg": tneg,
            }
        )
    return pd.DataFrame(rows)


def initiator_binary(chains: pd.DataFrame, lab: pd.DataFrame, cohort: list[str]) -> pd.DataFrame:
    """JA exam-init ≥4 chain as a presence/absence indicator vs severity."""
    from scipy.stats import mannwhitneyu
    from ados_ffm.data import TARGETS

    sub = chains[chains["task_id"].astype(int) == 3].copy()
    if sub.empty or "dyad_chain_ge4_per_min_exam_init" not in sub.columns:
        return pd.DataFrame()
    sub["participant_id"] = sub["participant_id"].astype(str)
    sub = sub.set_index("participant_id")
    ge4 = pd.to_numeric(sub["dyad_chain_ge4_per_min_exam_init"], errors="coerce")
    present = ge4 > 0.0
    y_by = {
        tgt["name"]: apply_target_y(lab.reindex([str(p) for p in cohort])[tgt["column"]])
        for tgt in TARGETS
        if tgt["name"] in ("SA", "CSS")
    }
    rows = []
    for tgt, y in y_by.items():
        y = y.copy()
        y.index = [str(i) for i in y.index]
        idx = sorted(set(present.index) & set(y.index))
        p = present.reindex(idx)
        yv = pd.to_numeric(y.reindex(idx), errors="coerce")
        m = p.notna() & np.isfinite(yv.to_numpy(dtype=float))
        p = p[m]
        yv = yv[m].to_numpy(dtype=float)
        pos = yv[p.to_numpy()]
        neg = yv[~p.to_numpy()]
        n1, n0 = int(pos.size), int(neg.size)
        if n1 < 3 or n0 < 3:
            u = p_mw = r_rb = float("nan")
        else:
            res = mannwhitneyu(pos, neg, alternative="two-sided")
            u = float(res.statistic)
            p_mw = float(res.pvalue)
            r_rb = float((2.0 * u) / (n1 * n0) - 1.0)
        rho = spearman_rho_fast(p.astype(float).to_numpy(), yv)
        rows.append(
            {
                "task": 3,
                "task_ja": TASK_JA[3],
                "who": "exam",
                "target": tgt,
                "n": int(n0 + n1),
                "n_present": n1,
                "n_absent": n0,
                "frac_present": float(n1 / (n0 + n1)) if (n0 + n1) else float("nan"),
                "rho_binary": float(rho),
                "U": u,
                "p_mannwhitney": p_mw,
                "rank_biserial": r_rb,
                "rank_biserial_lo": float("nan"),
                "rank_biserial_hi": float("nan"),
                "median_y_present": float(np.median(pos)) if n1 else float("nan"),
                "median_y_absent": float(np.median(neg)) if n0 else float("nan"),
            }
        )
        if n1 >= 3 and n0 >= 3:
            rng = np.random.default_rng(SEED_INIT + (0 if tgt == "SA" else 1))
            boots = []
            n = int(yv.size)
            p_arr = p.to_numpy()
            for _ in range(N_BOOT_INIT):
                ii = rng.integers(0, n, size=n)
                pb = p_arr[ii]
                yb = yv[ii]
                posb, negb = yb[pb], yb[~pb]
                if posb.size < 2 or negb.size < 2:
                    continue
                ub = float(mannwhitneyu(posb, negb, alternative="two-sided").statistic)
                boots.append((2.0 * ub) / (posb.size * negb.size) - 1.0)
            if len(boots) >= 20:
                lo, hi = np.quantile(boots, [0.025, 0.975])
                rows[-1]["rank_biserial_lo"] = float(lo)
                rows[-1]["rank_biserial_hi"] = float(hi)
    return pd.DataFrame(rows)


def ja_missingness(
    dyn: pd.DataFrame,
    lab: pd.DataFrame,
    cohort: list[str],
    *,
    feature: str = "dyad_chain_ge4_per_min",
    task: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Why JA drops from 59 to n=51, and whether those 8 differ on y."""
    from scipy.stats import mannwhitneyu
    from ados_extract.task_segments import load_task_segments_json, segments_for_participant
    from ados_ffm.data import TARGETS

    raw = load_task_segments_json()
    d = dyn[dyn["participant_id"].astype(str).isin(cohort)].copy()
    d["participant_id"] = d["participant_id"].astype(str)
    ja = d[d["task_id"].astype(int) == int(task)].set_index("participant_id")
    x = pd.to_numeric(ja[feature], errors="coerce") if feature in ja.columns else pd.Series(dtype=float)
    rows = []
    for pid in cohort:
        raw_rows = [r for r in raw.get(pid, []) if int(r.get("task_id", -1) or -1) == int(task)]
        # A task can be annotated as several stretches; show them all rather
        # than silently reporting the first one as if it were the interval.
        t0 = " / ".join(str(r.get("t0", "")).strip() for r in raw_rows)
        t1 = " / ".join(str(r.get("t1", "")).strip() for r in raw_rows)
        parsed = segments_for_participant(pid) or []
        has_parsed = any(int(s.get("task_id", -1)) == int(task) for s in parsed)
        has_row = pid in ja.index
        xval = float(x.loc[pid]) if has_row and pid in x.index else float("nan")
        dur = float(ja.loc[pid, "duration_sec"]) if has_row and "duration_sec" in ja.columns else float("nan")
        if not t0 or not t1:
            reason = "not_administered"
        elif not np.isfinite(xval):
            reason = "too_short_for_chain"
        else:
            reason = "in_analysis"
        rows.append(
            {
                "participant_id": pid,
                "task": int(task),
                "t0": t0,
                "t1": t1,
                "has_parsed_interval": bool(has_parsed),
                "has_feature_row": bool(has_row),
                "duration_sec": dur,
                "feature_finite": bool(np.isfinite(xval)),
                "reason": reason,
            }
        )
    ids = pd.DataFrame(rows)
    y_by = {
        tgt["name"]: apply_target_y(lab.reindex([str(p) for p in cohort])[tgt["column"]])
        for tgt in TARGETS
    }
    for y in y_by.values():
        y.index = [str(i) for i in y.index]
    miss_ids = ids.loc[ids["reason"] != "in_analysis", "participant_id"].tolist()
    ok_ids = ids.loc[ids["reason"] == "in_analysis", "participant_id"].tolist()
    cmp_rows = []
    for tgt, y in y_by.items():
        a = y.reindex(ok_ids).to_numpy(dtype=float)
        b = y.reindex(miss_ids).to_numpy(dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size >= 3 and b.size >= 3:
            res = mannwhitneyu(b, a, alternative="two-sided")
            u, p_mw = float(res.statistic), float(res.pvalue)
            r_rb = float((2.0 * u) / (a.size * b.size) - 1.0)
        else:
            u = p_mw = r_rb = float("nan")
        cmp_rows.append(
            {
                "target": tgt,
                "n_present": int(a.size),
                "n_missing": int(b.size),
                "median_present": float(np.median(a)) if a.size else float("nan"),
                "median_missing": float(np.median(b)) if b.size else float("nan"),
                "mean_present": float(np.mean(a)) if a.size else float("nan"),
                "mean_missing": float(np.mean(b)) if b.size else float("nan"),
                "U": u,
                "p_mannwhitney": p_mw,
                "rank_biserial_missing_gt_present": r_rb,
            }
        )
    return ids, pd.DataFrame(cmp_rows)


def all_pairwise_r(panel, *, min_n: int = 12) -> pd.DataFrame:
    """y-free Spearman r for every feature × every task pair. Unselected."""
    from ados_ffm.hetero import feature_meta

    tasks = [int(t) for t in panel.tasks]
    rows = []
    for feat in panel.features:
        xmap = panel.X[feat]
        meta = feature_meta(feat)
        for i, ta in enumerate(tasks):
            xa = xmap[int(ta)]
            for tb in tasks[i + 1 :]:
                xb = xmap[int(tb)]
                m = np.isfinite(xa) & np.isfinite(xb)
                n = int(m.sum())
                rho = spearman_rho_fast(xa[m], xb[m]) if n >= min_n else float("nan")
                rows.append(
                    {
                        "feature": feat,
                        **meta,
                        "task_a": int(ta),
                        "task_b": int(tb),
                        "n": n,
                        "r": float(rho),
                    }
                )
    return pd.DataFrame(rows)


def r_all_summary(r_df: pd.DataFrame) -> dict[str, Any]:
    r = r_df["r"].to_numpy(dtype=float)
    ok = np.isfinite(r)
    med = float(np.median(r[ok])) if ok.any() else float("nan")
    return {
        "n_pairs_nominal": int(len(r_df)),
        "n_pairs_valid": int(ok.sum()),
        "n_features": int(r_df["feature"].nunique()) if not r_df.empty else 0,
        "r_median": med,
        "r_q25": float(np.quantile(r[ok], 0.25)) if ok.any() else float("nan"),
        "r_q75": float(np.quantile(r[ok], 0.75)) if ok.any() else float("nan"),
        "r_mean": float(np.mean(r[ok])) if ok.any() else float("nan"),
        "lambda_star_at_median_r": float(lambda_star(med)) if np.isfinite(med) else float("nan"),
        "lambda_star_at_q25": float(lambda_star(float(np.quantile(r[ok], 0.25)))) if ok.any() else float("nan"),
        "lambda_star_at_q75": float(lambda_star(float(np.quantile(r[ok], 0.75)))) if ok.any() else float("nan"),
        "reliability_ceiling_sqrt_r": float(np.sqrt(max(med, 0.0))) if np.isfinite(med) else float("nan"),
        "frac_r_lt_0.2": float(np.mean(r[ok] < 0.2)) if ok.any() else float("nan"),
        "frac_r_lt_0": float(np.mean(r[ok] < 0.0)) if ok.any() else float("nan"),
    }


def initiator_verdict(rho: pd.DataFrame) -> dict[str, Any]:
    """Original prediction vs the sharper reading: the flip lives in exam-init chains."""
    if rho.empty:
        return {"executable": False}

    def _row(task: int, tgt: str, who: str) -> pd.Series | None:
        feat = (
            "dyad_chain_ge4_per_min"
            if who == "all"
            else f"dyad_chain_ge4_per_min_{who}_init"
        )
        hit = rho[(rho["task"] == task) & (rho["target"] == tgt) & (rho["feature"] == feat)]
        if hit.empty:
            return None
        return hit.iloc[0]

    def _pack(rec: pd.Series | None) -> dict[str, Any]:
        if rec is None:
            return {"rho": float("nan"), "lo": float("nan"), "hi": float("nan"), "covers0": None}
        lo = float(rec["rho_lo"]) if "rho_lo" in rec.index and pd.notna(rec["rho_lo"]) else float("nan")
        hi = float(rec["rho_hi"]) if "rho_hi" in rec.index and pd.notna(rec["rho_hi"]) else float("nan")
        return {
            "rho": float(rec["rho"]),
            "lo": lo,
            "hi": hi,
            "covers0": bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi),
            "excludes0": bool(np.isfinite(lo) and np.isfinite(hi) and (hi < 0.0 or lo > 0.0)),
        }

    out: dict[str, Any] = {
        "executable": True,
        "how": "speech_segments → exchange chains → first speaker of each chain",
        "prediction_original": "JA: exam-init carries +ρ; free play: child-init carries −ρ",
        "reading": "sign reversal lives inside examiner-initiated ≥4 chains; child-init is near 0 in both tasks",
    }
    for tgt in ("SA", "CSS"):
        ja_e, ja_c = _pack(_row(3, tgt, "exam")), _pack(_row(3, tgt, "child"))
        fr_e, fr_c = _pack(_row(7, tgt, "exam")), _pack(_row(7, tgt, "child"))
        ja_ok = (
            np.isfinite(ja_e["rho"])
            and ja_e["rho"] > 0
            and abs(ja_e["rho"]) > (abs(ja_c["rho"]) if np.isfinite(ja_c["rho"]) else 0.0)
        )
        fr_child_ok = (
            np.isfinite(fr_c["rho"])
            and fr_c["rho"] < 0
            and abs(fr_c["rho"]) > (abs(fr_e["rho"]) if np.isfinite(fr_e["rho"]) else 0.0)
        )
        exam_flip = (
            np.isfinite(ja_e["rho"])
            and np.isfinite(fr_e["rho"])
            and ja_e["rho"] > 0
            and fr_e["rho"] < 0
            and abs(ja_e["rho"]) > (abs(ja_c["rho"]) if np.isfinite(ja_c["rho"]) else 0.0)
            and abs(fr_e["rho"]) > (abs(fr_c["rho"]) if np.isfinite(fr_c["rho"]) else 0.0)
        )
        out[tgt] = {
            "JA_exam": ja_e,
            "JA_child": ja_c,
            "free_exam": fr_e,
            "free_child": fr_c,
            "JA_exam_carries_plus": bool(ja_ok),
            "free_child_carries_minus": bool(fr_child_ok),
            "exam_init_carries_the_flip": bool(exam_flip),
            "child_both_cover_0": bool(ja_c.get("covers0") and fr_c.get("covers0")),
        }
    out["JA_side_matches"] = bool(out["SA"]["JA_exam_carries_plus"] and out["CSS"]["JA_exam_carries_plus"])
    out["free_play_isolates_to_child"] = bool(
        out["SA"]["free_child_carries_minus"] and out["CSS"]["free_child_carries_minus"]
    )
    out["prediction_fully_supported"] = bool(out["JA_side_matches"] and out["free_play_isolates_to_child"])
    out["exam_init_carries_the_flip"] = bool(
        out["SA"]["exam_init_carries_the_flip"] and out["CSS"]["exam_init_carries_the_flip"]
    )
    out["child_near_zero"] = bool(out["SA"]["child_both_cover_0"] and out["CSS"]["child_both_cover_0"])
    return out


def _draw_r_hist(ax, r_df: pd.DataFrame, *, flip_r: np.ndarray | None = None) -> None:
    r = r_df["r"].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    ax.hist(r, bins=np.linspace(-0.6, 1.0, 33), color="#cfd4d6", edgecolor="white", zorder=1)
    med = float(np.median(r))
    star = float(np.sqrt(2.0 * (1.0 + med)) - 1.0)
    ax.axvline(med, color="#c0392b", lw=1.6, zorder=3, label=rf"median $r={med:.2f}$")
    ax.axvline(0.0, color="#888888", lw=0.7, ls="--", zorder=2)
    if flip_r is not None and np.isfinite(flip_r).any():
        ax.scatter(
            flip_r[np.isfinite(flip_r)],
            np.full(int(np.isfinite(flip_r).sum()), ax.get_ylim()[1] * 0.08 if ax.get_ylim()[1] else 1.0),
            s=28,
            c="#c0392b",
            marker="D",
            zorder=4,
            label="sign-flip pairs (8)",
        )
    ax.set_xlabel(r"$r=\mathrm{Corr}(x_A,x_B)$  (no $y$; all features × task pairs)")
    ax.set_ylabel("count")
    ax.set_title(rf"(a) Instrument: median $r$  ⇒  $\lambda^*={star:.2f}$")
    ax.legend(loc="upper left", fontsize=7)


def _draw_lambda_r(ax, pairs: pd.DataFrame) -> None:
    r_grid = np.linspace(-0.45, 1.0, 400)
    star = np.sqrt(2.0 * (1.0 + r_grid)) - 1.0
    ax.fill_between(r_grid, star, 1.0, color="#d9ead3", alpha=0.55, lw=0)
    ax.fill_between(r_grid, -1.0, star, color="#f4cccc", alpha=0.55, lw=0)
    ax.plot(r_grid, star, color="#333333", lw=1.3, label=r"$\lambda^*(r)$")
    ax.axhline(0.0, color="#666666", lw=0.7, ls="--")
    ax.axvline(0.0, color="#bbbbbb", lw=0.6)
    ax.axhline(np.sqrt(2.0) - 1.0, color="#888888", lw=0.6, ls=":", label=r"$\lambda^*(0)\approx0.41$")
    flip = pairs["sign_flip_cap"].astype(bool)
    ax.scatter(
        pairs.loc[~flip, "r_spear"],
        pairs.loc[~flip, "lambda"],
        s=40,
        c="#7a7a7a",
        marker="o",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="same sign",
    )
    ax.scatter(
        pairs.loc[flip, "r_spear"],
        pairs.loc[flip, "lambda"],
        s=52,
        c="#c0392b",
        marker="D",
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
        label="sign flip",
    )
    fid = (pairs["feature"] == "ges_child_fidgeting_per_min") & (pairs["target"] == "CSS")
    if fid.any():
        ax.scatter(
            pairs.loc[fid, "r_spear"],
            pairs.loc[fid, "lambda"],
            s=80,
            facecolors="none",
            edgecolors="#1e8449",
            linewidths=1.5,
            zorder=5,
            label="fidgeting",
        )
    ax.set_xlim(-0.5, 1.02)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\lambda=\rho_B/\rho_A$")
    ax.set_title("(b) Where the 21 pairs fell (point estimates; $\\lambda$ is a noisy ratio)")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.text(0.02, 0.96, "pool better", transform=ax.transAxes, va="top", fontsize=7, color="#276221")
    ax.text(0.02, 0.04, "pool worse", transform=ax.transAxes, va="bottom", fontsize=7, color="#990000")


def _draw_ci_forest(ax, verify: pd.DataFrame) -> None:
    n = len(verify)
    ys = np.arange(n)[::-1]
    task_en = {3: "JA", 6: "book", 7: "free", 9: "snack", 10: "rout"}
    labels = []
    for rec in verify.itertuples(index=False):
        short = str(rec.feature).replace("win_", "").replace("dyad_chain_", "chain_").replace("examiner_", "ex_")
        short = short.replace("_per_min", "").replace("_frac_delta", "d").replace("_rate_delta", "d")
        labels.append(
            f"{short} · {rec.target}  "
            f"{task_en.get(int(rec.task_pos), rec.task_pos)}/"
            f"{task_en.get(int(rec.task_neg), rec.task_neg)}"
        )
    ax.axvline(0.0, color="#888888", lw=0.9)
    pos = verify["rho_pos"].to_numpy(dtype=float)
    plo = verify["rho_pos_lo"].to_numpy(dtype=float)
    phi = verify["rho_pos_hi"].to_numpy(dtype=float)
    z = verify["rho_z_obs"].to_numpy(dtype=float)
    zlo = verify["rho_z_lo"].to_numpy(dtype=float)
    zhi = verify["rho_z_hi"].to_numpy(dtype=float)
    ax.errorbar(
        pos, ys + 0.16,
        xerr=np.vstack([pos - plo, phi - pos]),
        fmt="o", color="#2f6b4f", ecolor="#2f6b4f", elinewidth=1.0, capsize=2, ms=4,
        label=r"$\rho_+$",
    )
    ax.errorbar(
        z, ys - 0.16,
        xerr=np.vstack([z - zlo, zhi - z]),
        fmt="D", color="#0f5c6e", ecolor="#0f5c6e", elinewidth=1.0, capsize=2, ms=4,
        label=r"$\rho_z$",
    )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(r"Spearman $\rho$  ·  95% CI")
    ax.set_title("(c) Pooling moves the point to 0")
    ax.set_xlim(-0.72, 0.72)
    ax.legend(loc="lower right", fontsize=7)


def plot_exp3_main(r_df: pd.DataFrame, pairs: pd.DataFrame, verify: pd.DataFrame, path: Path) -> None:
    """(a) unselected r  (b) (λ,r) plane  (c) CI forest."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), gridspec_kw={"width_ratios": [1.05, 1.05, 1.35]})
    flip_r = pairs.loc[pairs["sign_flip_cap"].astype(bool), "r_spear"].to_numpy(dtype=float) if "sign_flip_cap" in pairs.columns else None
    _draw_r_hist(axes[0], r_df, flip_r=flip_r)
    _draw_lambda_r(axes[1], pairs)
    if not verify.empty:
        _draw_ci_forest(axes[2], verify)
    else:
        axes[2].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_r_hist(r_df: pd.DataFrame, pairs: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    flip_r = pairs.loc[pairs["sign_flip_cap"].astype(bool), "r_spear"].to_numpy(dtype=float)
    _draw_r_hist(ax, r_df, flip_r=flip_r)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lambda_r_data(pairs: pd.DataFrame, path: Path) -> None:
    """Single panel: where the 21 pairs fell. The curve is algebra, the points are data."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    _draw_lambda_r(ax, pairs)
    ax.set_title("Where the 21 pairs fell")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ci_forest(verify: pd.DataFrame, path: Path) -> None:
    """ρ+ vs ρ_z CIs. Non-trivial claim: pooling moves the point to 0 without shrinking the CI."""
    import matplotlib.pyplot as plt

    if verify.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    _draw_ci_forest(ax, verify)
    ax.set_title("Pooling moves the point to 0; the CI does not shrink")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_q_ranks(families: pd.DataFrame, ranks: pd.DataFrame, path: Path) -> None:
    """Unselected Q: p ranks. The curve is noise; marked points are data."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    fam = families[np.isfinite(families["p"])].copy()
    fam["p_rank"] = fam["p"].rank(method="min")
    ax.scatter(
        fam["p_rank"],
        -np.log10(np.clip(fam["p"].to_numpy(dtype=float), 1e-12, 1.0)),
        s=14,
        c="#cfd4d6",
        zorder=1,
        label="all 201 families",
    )
    colors = {"ja_free": "#c0392b", "speech_drift": "#0f5c6e", "other": "#1e8449"}
    markers = {"ja_free": "D", "speech_drift": "s", "other": "o"}
    seen = set()
    for rec in ranks.itertuples(index=False):
        phen = str(rec.phenomenon)
        lab = {
            "ja_free": "JA vs free play",
            "speech_drift": "speech drift",
            "other": "fidgeting",
        }.get(phen, phen)
        ax.scatter(
            [rec.p_rank],
            [-np.log10(max(float(rec.p), 1e-12))],
            s=64,
            c=colors.get(phen, "#333333"),
            marker=markers.get(phen, "o"),
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
            label=lab if phen not in seen else None,
        )
        seen.add(phen)
    n = int(np.isfinite(families["p"]).sum())
    k = np.arange(1, n + 1)
    ax.plot(k, -np.log10(k / (n + 1)), color="#888888", lw=0.8, ls="--", zorder=2, label="uniform order-stat")
    ax.set_xlabel("rank of Cochran Q p (201 families, unselected)")
    ax.set_ylabel(r"$-\log_{10} p$")
    ax.set_title("Heterogeneity is widespread, but no family survives BH($m$=201)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_duration(dur: pd.DataFrame, dyn: pd.DataFrame, cohort: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    d = dyn[dyn["participant_id"].astype(str).isin(cohort)]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    task_en = {
        1: "1 construct",
        2: "2 pretend",
        3: "3 JA",
        4: "4 demo",
        5: "5 picture",
        6: "6 book",
        7: "7 free play",
        8: "8 birthday",
        9: "9 snack",
        10: "10 routine",
    }
    data, labels = [], []
    for tid in TASKS:
        v = pd.to_numeric(d.loc[d["task_id"].astype(int) == tid, "duration_sec"], errors="coerce")
        v = v[np.isfinite(v)]
        data.append(v.to_numpy(dtype=float))
        labels.append(task_en[tid])
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    for i, tid in enumerate(TASKS, start=1):
        if tid in (3, 6, 7, 10):
            ax.get_xticklabels()[i - 1].set_fontweight("bold")
    ax.set_ylabel("duration (s)")
    ax.set_title("Task length (JA vs free play / book / routine)")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_initiator(rho: pd.DataFrame, path: Path) -> None:
    """JA vs free play: overall / examiner-initiated / child-initiated chain ρ with CI."""
    import matplotlib.pyplot as plt

    want = rho[rho["feature"].str.contains("ge4_per_min", regex=False)].copy()
    if want.empty:
        want = rho.copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), sharey=True)
    who_ord = ["all", "exam", "child"]
    who_lab = {"all": "all chains", "exam": "exam-init", "child": "child-init"}
    task_col = {3: "#c0392b", 7: "#0f5c6e"}
    task_lab = {3: "JA", 7: "free play"}
    for ax, tgt in zip(axes, ("SA", "CSS")):
        sub = want[want["target"] == tgt]
        x = np.arange(len(who_ord))
        width = 0.36
        for j, tid in enumerate((3, 7)):
            vals, yerr_lo, yerr_hi = [], [], []
            for w in who_ord:
                hit = sub[(sub["task"] == tid) & (sub["who"] == w)]
                if hit.empty:
                    vals.append(float("nan"))
                    yerr_lo.append(0.0)
                    yerr_hi.append(0.0)
                    continue
                v = float(hit["rho"].iloc[0])
                lo = float(hit["rho_lo"].iloc[0]) if "rho_lo" in hit.columns else float("nan")
                hi = float(hit["rho_hi"].iloc[0]) if "rho_hi" in hit.columns else float("nan")
                vals.append(v)
                yerr_lo.append(v - lo if np.isfinite(lo) and np.isfinite(v) else 0.0)
                yerr_hi.append(hi - v if np.isfinite(hi) and np.isfinite(v) else 0.0)
            xpos = x + (j - 0.5) * width
            ax.bar(
                xpos,
                vals,
                width=width,
                color=task_col[tid],
                label=task_lab[tid],
                yerr=np.vstack([yerr_lo, yerr_hi]),
                capsize=3,
                error_kw={"lw": 0.9, "ecolor": "#333333"},
            )
        ax.axhline(0.0, color="#888888", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([who_lab[w] for w in who_ord])
        ax.set_title(tgt)
        ax.set_ylim(-0.7, 0.75)
        ax.set_ylabel(r"Spearman $\rho$ with severity" if tgt == "SA" else "")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Sign reversal lives in examiner-initiated chains (bootstrap 95% CI)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_duration_y(tab: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    order = list(TASKS)
    tgts = ["SA", "RRB", "CSS"]
    mat = np.full((len(order), len(tgts)), np.nan)
    for rec in tab.itertuples(index=False):
        i = order.index(int(rec.task))
        j = tgts.index(str(rec.target))
        mat[i, j] = float(rec.rho)
    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    vmax = 0.45
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tgts)))
    ax.set_xticklabels(tgts)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([TASK_EN[t] for t in order])
    for i in range(len(order)):
        for j in range(len(tgts)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Spearman $\rho$(duration, $y$)")
    ax.set_title("Task duration vs severity (no features)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_r_vs_duration(tab: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    ok = tab[np.isfinite(tab["r"]) & np.isfinite(tab["min_dur_median"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(
        ok["min_dur_median"],
        ok["r"],
        s=8,
        c="#b8c0c4",
        alpha=0.35,
        linewidths=0,
        zorder=1,
        label="3015 pairs",
    )
    by_pair = ok.groupby(["task_a", "task_b"], as_index=False).agg(
        min_dur_median=("min_dur_median", "first"),
        r_median=("r", "median"),
    )
    ax.scatter(
        by_pair["min_dur_median"],
        by_pair["r_median"],
        s=36,
        c="#0f5c6e",
        zorder=3,
        label="median r per task pair",
        edgecolors="white",
        linewidths=0.4,
    )
    try:
        cats = pd.qcut(ok["min_dur_median"], 4, duplicates="drop")
        binned = ok.groupby(cats, observed=True).agg(
            x=("min_dur_median", "median"),
            y=("r", "median"),
        )
        ax.plot(binned["x"], binned["y"], color="#c0392b", lw=1.6, zorder=4, label="quartile medians of r")
    except ValueError:
        pass
    ax.axhline(float(np.median(ok["r"])), color="#666666", lw=0.8, ls="--", label="global median r")
    ax.set_xlabel(r"median of $\min(\mathrm{dur}_A,\mathrm{dur}_B)$ (s)")
    ax.set_ylabel(r"$r=\mathrm{Corr}(x_A,x_B)$")
    ax.set_title("Is low r just short intervals?")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lambda_unsel(cells: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    if cells.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(
        cells["r"],
        cells["lambda"],
        s=10,
        c=np.where(cells["sign_flip"], "#c0392b", "#7a7a7a"),
        alpha=0.35,
        linewidths=0,
        zorder=2,
    )
    r_grid = np.linspace(-0.45, 1.0, 400)
    ax.plot(r_grid, lambda_star(r_grid), color="#111111", lw=1.3, label=r"$\lambda^*(r)$")
    ax.axhline(0.0, color="#888888", lw=0.6)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\lambda=\rho_B/\rho_A$  (oriented $|\rho_A|\geq|\rho_B|$)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(rf"Unselected cells $|\rho_A|\geq{RHO_MIN_UNSEL}$: {100*float(cells['pool_worse'].mean()):.0f}% below $\lambda^*$")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lambda_perm(null: dict[str, Any], observed: float, path: Path) -> None:
    import matplotlib.pyplot as plt

    fracs = np.asarray(null.get("fracs") or [], dtype=float)
    fracs = fracs[np.isfinite(fracs)]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if fracs.size:
        ax.hist(fracs, bins=20, color="#cfd4d6", edgecolor="white")
        ax.axvline(
            float(null["null_median"]),
            color="#666666",
            lw=1.2,
            ls="--",
            label=rf"null median {null['null_median']:.2f}",
        )
        ax.axvline(
            float(null["null_q95"]),
            color="#888888",
            lw=0.8,
            ls=":",
            label=rf"null 95% {null['null_q95']:.2f}",
        )
    ax.axvline(observed, color="#c0392b", lw=1.6, label=rf"observed {observed:.2f}")
    ax.set_xlabel(r"fraction $\lambda<\lambda^*$  ($|\rho_A|\geq 0.2$, oriented)")
    ax.set_ylabel("permutations")
    ax.set_title("Pool-worse fraction vs subject-level y shuffle")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def forest_summary(verify: pd.DataFrame) -> dict[str, Any]:
    if verify.empty:
        return {}
    w_pos = (verify["rho_pos_hi"] - verify["rho_pos_lo"]).to_numpy(dtype=float)
    w_z = (verify["rho_z_hi"] - verify["rho_z_lo"]).to_numpy(dtype=float)
    hit = verify[(verify["feature"] == "dyad_chain_ge4_per_min") & (verify["target"] == "CSS")]
    out = {
        "median_ci_width_rho_plus": float(np.nanmedian(w_pos)),
        "median_ci_width_rho_z": float(np.nanmedian(w_z)),
        "n_rho_z_covers_0": int(verify["rho_z_covers_0"].sum()),
        "n_rho_plus_excludes_0": int(verify["rho_pos_excludes_0"].sum()),
        "n_pairs": int(len(verify)),
    }
    if not hit.empty:
        r = hit.iloc[0]
        out["chain_ge4_css_rho_plus"] = float(r["rho_pos"])
        out["chain_ge4_css_rho_plus_lo"] = float(r["rho_pos_lo"])
        out["chain_ge4_css_rho_plus_hi"] = float(r["rho_pos_hi"])
        out["chain_ge4_css_rho_z"] = float(r["rho_z_obs"])
        out["chain_ge4_css_rho_z_lo"] = float(r["rho_z_lo"])
        out["chain_ge4_css_rho_z_hi"] = float(r["rho_z_hi"])
        out["chain_ge4_css_width_plus"] = float(r["rho_pos_hi"] - r["rho_pos_lo"])
        out["chain_ge4_css_width_z"] = float(r["rho_z_hi"] - r["rho_z_lo"])
    return out


def r_summary(pairs: pd.DataFrame) -> dict[str, Any]:
    flip = pairs["sign_flip_cap"].astype(bool)
    r_f = pairs.loc[flip, "r_spear"].to_numpy(dtype=float)
    r_s = pairs.loc[~flip, "r_spear"].to_numpy(dtype=float)
    fid = (pairs["feature"] == "ges_child_fidgeting_per_min") & (pairs["target"] == "CSS")
    out: dict[str, Any] = {
        "n_flip": int(flip.sum()),
        "n_same": int((~flip).sum()),
        "r_flip": [float(x) for x in np.sort(r_f)] if r_f.size else [],
        "r_flip_median": float(np.median(r_f)) if r_f.size else float("nan"),
        "r_flip_min": float(np.min(r_f)) if r_f.size else float("nan"),
        "r_flip_max": float(np.max(r_f)) if r_f.size else float("nan"),
        "r_same_median": float(np.median(r_s)) if r_s.size else float("nan"),
        "r_same_q25": float(np.quantile(r_s, 0.25)) if r_s.size else float("nan"),
        "r_same_q75": float(np.quantile(r_s, 0.75)) if r_s.size else float("nan"),
        "n_pool_worse": int(pairs["pool_worse"].sum()),
        "n_red_and_same_sign": int(((~flip) & pairs["pool_worse"].astype(bool)).sum()),
        "lambda_star_at_0": float(np.sqrt(2.0) - 1.0),
    }
    if fid.any():
        r = pairs.loc[fid].iloc[0]
        out["fidgeting_r"] = float(r["r_spear"])
        out["fidgeting_lambda"] = float(r["lambda"])
        out["fidgeting_lambda_star"] = float(r["lambda_star"])
        out["fidgeting_gain"] = float(r["p1_gain_obs"]) if "p1_gain_obs" in pairs.columns else float("nan")
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run(
    out: Path,
    *,
    features: Path | None = None,
    windows: Path | None = None,
    hits: Path | None = None,
    n_boot_maxstat: int = 4999,
    n_perm_lambda: int = N_PERM_LAM,
    skip_initiators: bool = False,
    skip_maxstat: bool = True,
    reextract_initiators: bool = False,
) -> dict[str, Any]:
    """Write non-tautological Exp3 tables and figures into `out`."""
    from ados_ffm.data import DEFAULT_FEATURES, DEFAULT_WINDOWS, load_features
    from ados_ffm.hetero import (
        N_BOOT_PAIR,
        SEED_PAIR,
        build_panel,
        exp1_multitask_pairs,
        family_seed,
        maxstat_p,
    )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    feat_path = Path(features) if features else DEFAULT_FEATURES
    win_path = Path(windows) if windows else DEFAULT_WINDOWS
    hits_path = Path(hits) if hits else ROOT / "outputs/exp1/hits_main.csv"

    pairs = pd.read_csv(out / "pairs_21.csv")
    verify = pd.read_csv(out / "pool_verify_10.csv") if (out / "pool_verify_10.csv").exists() else pd.DataFrame()
    families = pd.read_csv(out / "families.csv") if (out / "families.csv").exists() else pd.DataFrame()

    cohort = load_cohort()
    lab = load_labels()
    dyn = load_features(feat_path)
    win = pd.read_csv(win_path)
    win["participant_id"] = win["participant_id"].astype(str)
    panel = build_panel(dyn, win, lab, cohort, TASKS)

    print(f"all-pair r  features={len(panel.features)}  pairs/feature={len(TASKS)*(len(TASKS)-1)//2}", flush=True)
    r_all = all_pairwise_r(panel)
    r_all.to_csv(out / "r_all_pairs.csv", index=False)
    r_all_sum = r_all_summary(r_all)
    print(
        f"  valid {r_all_sum['n_pairs_valid']}/{r_all_sum['n_pairs_nominal']}  "
        f"median r={r_all_sum['r_median']:.3f}  "
        f"λ*(median)={r_all_sum['lambda_star_at_median_r']:.3f}  "
        f"√r ceiling={r_all_sum['reliability_ceiling_sqrt_r']:.3f}",
        flush=True,
    )
    plot_r_hist(r_all, pairs, out / "fig_r_hist.png")
    plot_lambda_r_data(pairs, out / "fig_lambda_r.png")
    if not verify.empty:
        plot_ci_forest(verify, out / "fig_ci_forest.png")
    plot_exp3_main(r_all, pairs, verify, out / "fig_exp3_main.png")

    ranks = pd.DataFrame()
    q_curve: dict[str, Any] = {}
    if not families.empty:
        ranks = q_ranks(families)
        ranks.to_csv(out / "q_ranks.csv", index=False)
        plot_q_ranks(families, ranks, out / "fig_q_ranks.png")
        q_curve = q_p_curve_summary(families)
        print("Q ranks", ranks[["feature", "target", "p_rank", "p"]].to_string(index=False), flush=True)
        print(
            "Q p-curve  "
            + "  ".join(
                f"p<{e['threshold']:.2f}: {e['n_obs']}/{q_curve['n_families']} (exp {e['n_expected']:.1f})"
                for e in q_curve.get("ecdf") or []
            ),
            flush=True,
        )

    dur = duration_by_task(dyn, cohort)
    dur.to_csv(out / "duration_by_task.csv", index=False)
    plot_duration(dur, dyn, cohort, out / "fig_duration.png")
    print("duration medians (s)", dur[["task", "task_ja", "median_sec"]].to_string(index=False), flush=True)

    dur_y = duration_y_correlations(dyn, lab, cohort)
    dur_y.to_csv(out / "duration_y.csv", index=False)
    plot_duration_y(dur_y, out / "fig_duration_y.png")
    print("duration vs y", dur_y.to_string(index=False), flush=True)

    miss_ids, miss_y = ja_missingness(dyn, lab, cohort)
    miss_ids.to_csv(out / "ja_missing_ids.csv", index=False)
    miss_y.to_csv(out / "ja_missing_y.csv", index=False)
    print("JA missingness", miss_ids["reason"].value_counts().to_string(), flush=True)
    print(miss_y.to_string(index=False), flush=True)

    wide = duration_wide(dyn, cohort)
    pair_dur = pair_min_duration(wide)
    pair_dur.to_csv(out / "pair_min_duration.csv", index=False)
    r_dur = r_vs_duration_table(r_all, pair_dur)
    r_dur_sum = r_vs_duration_summary(r_dur)
    plot_r_vs_duration(r_dur, out / "fig_r_vs_dur.png")
    print(
        f"r vs min-dur  spearman={r_dur_sum.get('spearman_r_vs_min_dur')}  "
        f"pair-level={r_dur_sum.get('pair_level_spearman')}  "
        f"quartile medians={r_dur_sum.get('bin_medians')}",
        flush=True,
    )

    flips8 = pairs21_as_flips(pairs)
    part = duration_partial_table(dyn, win, lab, flips8, cohort)
    part.to_csv(out / "duration_partial.csv", index=False)
    n_ok = int(part["sign_survives"].sum()) if not part.empty else 0
    n_rate = int(part["rate_normalized"].sum()) if "rate_normalized" in part.columns and not part.empty else 0
    n_norate_ok = (
        int((~part["rate_normalized"] & part["sign_survives"]).sum())
        if "rate_normalized" in part.columns and not part.empty
        else 0
    )
    n_norate = int((~part["rate_normalized"]).sum()) if "rate_normalized" in part.columns and not part.empty else 0
    print(
        f"duration-partial  8 pairs × 2 sides  sign survives {n_ok}/{len(part)}  "
        f"non-rate {n_norate_ok}/{n_norate}  rate {n_ok - n_norate_ok}/{n_rate}",
        flush=True,
    )
    print(part.to_string(index=False), flush=True)

    print("unselected λ vs λ*", flush=True)
    lam_cells = unselected_lambda_cells(panel, r_all)
    lam_cells.to_csv(out / "lambda_unselected.csv", index=False)
    lam_sum = unselected_lambda_summary(lam_cells)
    plot_lambda_unsel(lam_cells, out / "fig_lambda_unsel.png")
    print(
        f"  |ρ_A|≥{RHO_MIN_UNSEL} cells={lam_sum.get('n_cells')}  "
        f"pool worse {lam_sum.get('n_pool_worse')}/{lam_sum.get('n_cells')} "
        f"({100*float(lam_sum.get('frac_pool_worse') or 0):.0f}%)  "
        f"same-sign worse {100*float(lam_sum.get('frac_pool_worse_same_sign') or 0):.0f}%",
        flush=True,
    )
    perm: dict[str, Any] = {}
    if n_perm_lambda and n_perm_lambda > 0:
        print(f"λ-perm  B={n_perm_lambda}  subject-level y shuffle", flush=True)
        perm = permute_lambda_frac(panel, r_all, n_perm=n_perm_lambda)
        obs = float(lam_sum.get("frac_pool_worse") or float("nan"))
        obs_same = float(lam_sum.get("frac_pool_worse_same_sign") or float("nan"))
        n_ge = int(sum(1 for x in perm.get("fracs") or [] if np.isfinite(x) and x >= obs - 1e-12))
        n_ge_s = int(
            sum(1 for x in perm.get("fracs_same") or [] if np.isfinite(x) and x >= obs_same - 1e-12)
        )
        perm["observed"] = obs
        perm["observed_same"] = obs_same
        perm["p_perm"] = (n_ge + 1) / (int(perm["n_perm"]) + 1)
        perm["p_perm_same"] = (n_ge_s + 1) / (int(perm["n_perm"]) + 1)
        perm["distinguishable"] = bool(np.isfinite(obs) and obs > float(perm["null_q95"]))
        pd.DataFrame({"frac": perm["fracs"], "frac_same": perm["fracs_same"]}).to_csv(
            out / "lambda_perm.csv", index=False
        )
        plot_lambda_perm(perm, obs, out / "fig_lambda_perm.png")
        print(
            f"  observed {obs:.3f}  null median {perm['null_median']:.3f}  "
            f"95% {perm['null_q95']:.3f}  p_perm={perm['p_perm']:.3f}  "
            f"same-sign obs {obs_same:.3f} vs null {perm['null_same_median']:.3f}",
            flush=True,
        )
        perm_meta = {k: v for k, v in perm.items() if k not in ("fracs", "fracs_same")}
    else:
        prev = {}
        if (out / "findings_meta.json").exists():
            prev = json.loads((out / "findings_meta.json").read_text(encoding="utf-8")).get("lambda_perm") or {}
        perm_meta = prev

    chain_path = out / "chain_init.csv"
    if skip_initiators:
        chains = pd.read_csv(chain_path) if chain_path.exists() else pd.DataFrame()
    elif chain_path.exists() and not reextract_initiators:
        chains = pd.read_csv(chain_path)
        print(f"chain-init reused  {chain_path}", flush=True)
    else:
        print("chain-init extract", flush=True)
        chains = extract_chain_initiators(cohort)
        chains.to_csv(chain_path, index=False)
    init_rho = pd.DataFrame()
    init_n = pd.DataFrame()
    init_bin = pd.DataFrame()
    if not chains.empty:
        init_rho = initiator_correlations(chains, lab, cohort)
        init_rho.to_csv(out / "initiator_rho.csv", index=False)
        init_n = initiator_counts(chains)
        init_n.to_csv(out / "initiator_counts.csv", index=False)
        init_bin = initiator_binary(chains, lab, cohort)
        if not init_bin.empty:
            init_bin.to_csv(out / "initiator_binary.csv", index=False)
            print("initiator binary (JA exam-init ≥4 present)", init_bin.to_string(index=False), flush=True)
        plot_initiator(init_rho, out / "fig_initiator.png")
        print("initiator ρ", init_rho.to_string(index=False), flush=True)
        print("initiator counts", init_n.to_string(index=False), flush=True)
    mech = initiator_verdict(init_rho)
    print(
        f"initiator check  executable={mech.get('executable')}  "
        f"exam-init carries flip={mech.get('exam_init_carries_the_flip')}  "
        f"child CI covers 0={mech.get('child_near_zero')}  "
        f"original prediction fully supported={mech.get('prediction_fully_supported')}",
        flush=True,
    )

    maxstat_rows = []
    if not skip_maxstat and hits_path.exists():
        hits_df = pd.read_csv(hits_path)
        multi = exp1_multitask_pairs(hits_df)
        n_boot = n_boot_maxstat or N_BOOT_PAIR
        print(f"max-stat null  families with ≥3 Exp1-hit tasks  B={n_boot}", flush=True)
        for rec in multi.itertuples(index=False):
            if int(rec.n_hit_tasks) < 3:
                continue
            tids = [int(t[0]) for t in rec.tasks]
            ms = maxstat_p(
                panel.X[rec.feature],
                panel.y[rec.target],
                tids,
                n_boot=n_boot,
                seed=family_seed(rec.feature, rec.target, SEED_PAIR) + 101,
            )
            maxstat_rows.append(
                {
                    "feature": rec.feature,
                    "target": rec.target,
                    "sign_flip": bool(rec.sign_flip),
                    **ms,
                }
            )
            print(
                f"  {rec.feature}×{rec.target}  hits={rec.n_hit_tasks}  "
                f"p_maxstat={ms.get('p_maxstat')}",
                flush=True,
            )
    maxstat = pd.DataFrame(maxstat_rows)
    if not maxstat.empty:
        maxstat.to_csv(out / "maxstat_3hit.csv", index=False)

    def _dur_y_rho(task: int, tgt: str) -> float:
        hit = dur_y[(dur_y["task"] == task) & (dur_y["target"] == tgt)]
        return float(hit["rho"].iloc[0]) if not hit.empty else float("nan")

    summary = {
        **r_summary(pairs),
        **forest_summary(verify),
        **r_all_sum,
        **q_curve,
        "q_min_p": float(families["p"].min()) if not families.empty else None,
        "q_min_p_rank1_feature": (
            str(families.loc[families["p"].idxmin(), "feature"]) if not families.empty else None
        ),
        "q_expected_min_uniform": 1.0 / (1 + int(len(families))) if not families.empty else None,
        "duration_partial_sign_survives": n_ok,
        "duration_partial_n": int(len(part)),
        "duration_y": {
            "max_abs": float(dur_y["rho"].abs().max()) if not dur_y.empty else None,
            "median_abs": float(dur_y["rho"].abs().median()) if not dur_y.empty else None,
            "JA_SA": _dur_y_rho(3, "SA"),
            "JA_RRB": _dur_y_rho(3, "RRB"),
            "JA_CSS": _dur_y_rho(3, "CSS"),
            "free_SA": _dur_y_rho(7, "SA"),
            "free_RRB": _dur_y_rho(7, "RRB"),
            "free_CSS": _dur_y_rho(7, "CSS"),
        },
        "r_vs_duration": r_dur_sum,
        "lambda_unselected": lam_sum,
        "lambda_perm": perm_meta,
        "duration_partial_nonrate_survives": n_norate_ok,
        "duration_partial_nonrate_n": n_norate,
        "n_maxstat_families": int(len(maxstat)),
        "no_hypothesis_tests": True,
        "initiator": mech,
        "initiator_counts": init_n.to_dict(orient="records") if not init_n.empty else [],
        "initiator_binary": init_bin.to_dict(orient="records") if not init_bin.empty else [],
        "ja_missing": {
            "n_not_administered": int((miss_ids["reason"] == "not_administered").sum()),
            "n_too_short": int((miss_ids["reason"] == "too_short_for_chain").sum()),
            "n_in_analysis": int((miss_ids["reason"] == "in_analysis").sum()),
            "y": miss_y.to_dict(orient="records"),
        },
    }
    (out / "findings_meta.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # drop the tautological identity figure if a previous run left it
    stale = out / "fig_eq3_verify.png"
    if stale.exists():
        stale.unlink()
    print(f"findings wrote {out}", flush=True)
    return summary

