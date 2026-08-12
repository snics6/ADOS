#!/usr/bin/env python
"""Stability selection for within-fold Stage1 notable cells.

Repeatedly subsample participants (default 80%), re-run Stage1 scoring +
notable selection from scratch, and record how often each cell is selected.

    python scripts/run_stability_selection.py --experiment items
    python scripts/run_stability_selection.py --experiment aggregates
    python scripts/run_stability_selection.py --experiment both

Meinshausen & Bühlmann (2010)-style reporting: selection frequency ∈ [0,1].
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ados_seg.data.tasks import TASK_NAMES_JA  # noqa: E402
from ados_seg.dynamics.extract import read_dynamics  # noqa: E402
from ados_seg.dynamics.scale_adjust import ScaleAdjuster  # noqa: E402
from run_diagnosis_by_role import BLOCKS, CV_BY_TASK, TASKS  # noqa: E402
from run_within_fold_aggregates import (  # noqa: E402
    AUC_MIN as AGG_AUC_MIN,
)
from run_within_fold_aggregates import (  # noqa: E402
    R_MIN,
    TARGETS,
    TOP_K as AGG_TOP_K,
    cell_xy as agg_cell_xy,
    detect_notable as agg_detect_notable,
    run_score as agg_run_score,
)
from run_within_fold_items import (  # noqa: E402
    METHODS,
    cell_xy as item_cell_xy,
    detect_notable as item_detect_notable,
    items_of,
    load_labels,
    run_score_job as item_run_score,
)

DEFAULT_OUT = ROOT / "outputs/experiment/stability_selection"


def _pids_for_task(feat: pd.DataFrame, task: int, cols: list[str]) -> np.ndarray:
    sub = feat[feat.task_id == task][["participant_id", *cols]].dropna(subset=cols)
    return sub.participant_id.astype(str).to_numpy()


def build_item_cache(feat, lab, items):
    """(task, role, item) -> X, y, coh, pids, cols."""
    cache = {}
    for task in TASKS:
        for role, cols in BLOCKS.items():
            pids_task = None
            for item in items:
                got = item_cell_xy(feat, lab, task, cols, item)
                if got is None:
                    continue
                X, y, coh = got
                n_splits, _ = CV_BY_TASK[task]
                if min(y.sum(), len(y) - y.sum()) < n_splits:
                    continue
                # rebuild pids aligned with cell_xy row order
                sub = feat[feat.task_id == task][["participant_id", *cols]].dropna(
                    subset=cols
                )
                raw = pd.to_numeric(
                    lab.reindex(sub.participant_id.astype(str))[item], errors="coerce"
                )
                raw = raw.where(raw != 8)
                yy = (raw > 0).astype(float).where(raw.notna())
                m = yy.notna().to_numpy()
                pids = sub.participant_id.astype(str).to_numpy()[m]
                # drop rows that fail minority filter already applied in cell_xy
                if len(pids) != len(y):
                    # fallback: match by recomputing mask with same filters
                    yv = yy.loc[m].to_numpy(float)
                    keep = np.ones(len(yv), dtype=bool)
                    if min(yv.sum(), len(yv) - yv.sum()) < 3:
                        continue
                    pids = pids
                assert len(pids) == len(y), (len(pids), len(y), item, task, role)
                cache[(task, role, item)] = (X, y, coh, pids, cols)
    return cache


def build_agg_cache(feat, lab):
    cache = {}
    for task in TASKS:
        for role, cols in BLOCKS.items():
            for tid, mode, key in TARGETS:
                got = agg_cell_xy(feat, lab, task, cols, key, mode)
                if got is None:
                    continue
                X, y, coh = got
                n_splits, _ = CV_BY_TASK[task]
                if mode == "binary" and min(y.sum(), len(y) - y.sum()) < n_splits:
                    continue
                # rebuild pids like agg_cell_xy
                from run_within_fold_aggregates import make_y

                sub = feat[feat.task_id == task][["participant_id", *cols]].dropna(
                    subset=cols
                )
                keep_idx = []
                for i, pid in enumerate(sub.participant_id.astype(str)):
                    if pid not in lab.index:
                        continue
                    v = make_y(lab.loc[pid], key, mode)
                    if v is None or not np.isfinite(v):
                        continue
                    keep_idx.append(i)
                pids = sub.participant_id.astype(str).to_numpy()[keep_idx]
                assert len(pids) == len(y), (len(pids), len(y), tid, task, role)
                cache[(task, role, tid, mode)] = (X, y, coh, pids, cols)
    return cache


def _mask_usable_binary(y: np.ndarray, n_splits: int) -> bool:
    if len(y) < 12:
        return False
    mino = min(y.sum(), len(y) - y.sum())
    if mino < 3 or mino < n_splits:
        return False
    return True


def _mask_usable_reg(y: np.ndarray) -> bool:
    return len(y) >= 12 and float(np.std(y)) > 1e-12


def _score_item_fit(X, y, coh, n_splits, n_repeats, seed, method):
    warnings.filterwarnings("ignore")
    try:
        return item_run_score(X, y, coh, n_splits, n_repeats, seed, method)
    except Exception:
        return None


def _score_agg_fit(X, y, coh, n_splits, n_repeats, seed, method, binary):
    warnings.filterwarnings("ignore")
    try:
        return agg_run_score(X, y, coh, n_splits, n_repeats, seed, method, binary)
    except Exception:
        return None


def run_items_stability(
    cache: dict,
    all_pids: np.ndarray,
    *,
    n_resamples: int,
    frac: float,
    top_k: int,
    auc_min: float,
    jobs: int,
    seed: int,
    cv_repeat_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_keep = max(12, int(round(frac * len(all_pids))))
    cell_keys = list(cache.keys())

    # eligibility / selection counters
    eligible_counts = Counter()
    selected_counts = Counter()
    selected_methods = defaultdict(Counter)
    # also store full-sample reference scores later

    t0 = time.time()
    for b in range(n_resamples):
        chosen = set(rng.choice(all_pids, size=n_keep, replace=False).tolist())
        jobs_list = []
        meta = []
        for (task, role, item) in cell_keys:
            X, y, coh, pids, cols = cache[(task, role, item)]
            m = np.array([p in chosen for p in pids], dtype=bool)
            if m.sum() < 12:
                continue
            Xs, ys, cs = X[m], y[m], coh[m]
            n_splits, n_repeats = CV_BY_TASK[task]
            n_repeats = max(2, int(round(n_repeats * cv_repeat_scale)))
            if not _mask_usable_binary(ys, n_splits):
                continue
            eligible_counts[(task, role, item)] += 1
            for mi, method in enumerate(METHODS):
                s = (
                    900_000
                    + b * 10_000
                    + task * 100
                    + {"child": 1, "examiner": 2, "dyad": 3}[role] * 10
                    + (hash(item) % 97)
                    + mi
                )
                jobs_list.append(
                    delayed(_score_item_fit)(
                        Xs, ys, cs, n_splits, n_repeats, s, method
                    )
                )
                meta.append((task, role, item, method, int(m.sum()), int(min(ys.sum(), len(ys) - ys.sum()))))

        if not jobs_list:
            continue
        scores = Parallel(n_jobs=jobs, verbose=0)(jobs_list)
        rows = []
        for (task, role, item, method, n, minority), sc in zip(meta, scores):
            if sc is None:
                continue
            rows.append(
                dict(
                    task=task,
                    task_ja=TASK_NAMES_JA[task],
                    role=role,
                    item=item,
                    method=method,
                    n=n,
                    minority=minority,
                    auc=sc["auc"],
                    bac=sc.get("bac", np.nan),
                    status="ok",
                )
            )
        if not rows:
            continue
        stage1 = pd.DataFrame(rows)
        notable = item_detect_notable(stage1, top_k=top_k, auc_min=auc_min)
        for _, r in notable.iterrows():
            key = (int(r["task"]), str(r["role"]), str(r["item"]))
            selected_counts[key] += 1
            selected_methods[key][str(r["method"])] += 1

        if (b + 1) % 10 == 0 or b == 0:
            elapsed = time.time() - t0
            rate = (b + 1) / elapsed
            eta = (n_resamples - b - 1) / max(rate, 1e-9)
            print(
                f"[items] resample {b+1}/{n_resamples}  "
                f"fits={len(jobs_list)}  selected={len(notable)}  "
                f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
                flush=True,
            )

    # assemble frequency table over all Stage1 cells
    rows = []
    for (task, role, item) in cell_keys:
        key = (task, role, item)
        n_elig = eligible_counts[key]
        n_sel = selected_counts[key]
        meth = selected_methods[key]
        top_meth = meth.most_common(1)[0][0] if meth else ""
        rows.append(
            dict(
                task=task,
                task_ja=TASK_NAMES_JA[task],
                role=role,
                item=item,
                n_full=len(cache[key][1]),
                n_resamples=n_resamples,
                n_eligible=n_elig,
                n_selected=n_sel,
                freq=n_sel / n_resamples,
                freq_given_eligible=(n_sel / n_elig) if n_elig else 0.0,
                top_method=top_meth,
            )
        )
    freq = pd.DataFrame(rows).sort_values(
        ["freq", "freq_given_eligible"], ascending=False
    )
    return freq, pd.DataFrame(
        [
            {
                "resample_note": "counts aggregated; per-resample notables not stored",
                "n_resamples": n_resamples,
                "frac": frac,
                "n_keep": n_keep,
                "top_k": top_k,
                "auc_min": auc_min,
            }
        ]
    )


def run_agg_stability(
    cache: dict,
    all_pids: np.ndarray,
    *,
    n_resamples: int,
    frac: float,
    jobs: int,
    seed: int,
    cv_repeat_scale: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_keep = max(12, int(round(frac * len(all_pids))))
    cell_keys = list(cache.keys())
    eligible_counts = Counter()
    selected_counts = Counter()
    selected_methods = defaultdict(Counter)

    t0 = time.time()
    for b in range(n_resamples):
        chosen = set(rng.choice(all_pids, size=n_keep, replace=False).tolist())
        jobs_list = []
        meta = []
        for (task, role, tid, mode) in cell_keys:
            X, y, coh, pids, cols = cache[(task, role, tid, mode)]
            m = np.array([p in chosen for p in pids], dtype=bool)
            if m.sum() < 12:
                continue
            Xs, ys, cs = X[m], y[m], coh[m]
            n_splits, n_repeats = CV_BY_TASK[task]
            n_repeats = max(2, int(round(n_repeats * cv_repeat_scale)))
            binary = mode == "binary"
            if binary and not _mask_usable_binary(ys, n_splits):
                continue
            if (not binary) and not _mask_usable_reg(ys):
                continue
            eligible_counts[(task, role, tid, mode)] += 1
            for mi, method in enumerate(METHODS):
                s = (
                    800_000
                    + b * 10_000
                    + task * 100
                    + {"child": 1, "examiner": 2, "dyad": 3}[role] * 10
                    + (hash(tid) % 97)
                    + mi
                )
                jobs_list.append(
                    delayed(_score_agg_fit)(
                        Xs, ys, cs, n_splits, n_repeats, s, method, binary
                    )
                )
                meta.append(
                    (
                        task,
                        role,
                        tid,
                        mode,
                        method,
                        int(m.sum()),
                        int(min(ys.sum(), len(ys) - ys.sum())) if binary else -1,
                    )
                )

        if not jobs_list:
            continue
        scores = Parallel(n_jobs=jobs, verbose=0)(jobs_list)
        rows = []
        for (task, role, tid, mode, method, n, minority), sc in zip(meta, scores):
            if sc is None:
                continue
            row = dict(
                task=task,
                task_ja=TASK_NAMES_JA[task],
                role=role,
                target=tid,
                mode=mode,
                method=method,
                n=n,
                minority=minority if minority >= 0 else np.nan,
                status="ok",
                auc=sc.get("auc", np.nan),
                bac=sc.get("bac", np.nan),
                r=sc.get("r", np.nan),
                ccc=sc.get("ccc", np.nan),
            )
            rows.append(row)
        if not rows:
            continue
        stage1 = pd.DataFrame(rows)
        notables = []
        if (stage1["mode"] == "binary").any():
            notables.append(agg_detect_notable(stage1[stage1["mode"] == "binary"], True))
        if (stage1["mode"] == "regression").any():
            notables.append(
                agg_detect_notable(stage1[stage1["mode"] == "regression"], False)
            )
        if not notables:
            continue
        notable = pd.concat(notables, ignore_index=True)
        for _, r in notable.iterrows():
            key = (int(r["task"]), str(r["role"]), str(r["target"]), str(r["mode"]))
            selected_counts[key] += 1
            selected_methods[key][str(r["method"])] += 1

        if (b + 1) % 10 == 0 or b == 0:
            elapsed = time.time() - t0
            rate = (b + 1) / elapsed
            eta = (n_resamples - b - 1) / max(rate, 1e-9)
            print(
                f"[aggregates] resample {b+1}/{n_resamples}  "
                f"fits={len(jobs_list)}  selected={len(notable)}  "
                f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
                flush=True,
            )

    rows = []
    for (task, role, tid, mode) in cell_keys:
        key = (task, role, tid, mode)
        n_elig = eligible_counts[key]
        n_sel = selected_counts[key]
        meth = selected_methods[key]
        top_meth = meth.most_common(1)[0][0] if meth else ""
        rows.append(
            dict(
                task=task,
                task_ja=TASK_NAMES_JA[task],
                role=role,
                target=tid,
                mode=mode,
                n_full=len(cache[key][1]),
                n_resamples=n_resamples,
                n_eligible=n_elig,
                n_selected=n_sel,
                freq=n_sel / n_resamples,
                freq_given_eligible=(n_sel / n_elig) if n_elig else 0.0,
                top_method=top_meth,
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["freq", "freq_given_eligible"], ascending=False
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment",
        choices=("items", "aggregates", "both"),
        default="both",
    )
    ap.add_argument("--n-resamples", type=int, default=200)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument(
        "--cv-repeat-scale",
        type=float,
        default=1.0,
        help="Scale CV n_repeats (1.0 = same as Stage1; 0.5 ≈ half cost)",
    )
    ap.add_argument("--top-k-items", type=int, default=20)
    ap.add_argument("--auc-min-items", type=float, default=0.65)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    args.out.mkdir(parents=True, exist_ok=True)

    feat = ScaleAdjuster().fit_transform(
        read_dynamics(ROOT / "outputs/features/dyn_v1/features_task.csv")
    )
    lab = load_labels()
    lab = lab.loc[lab.index.intersection(feat.participant_id.astype(str).unique())]
    all_pids = np.array(sorted(lab.index.astype(str).unique()))
    print(f"participants: {len(all_pids)}  keep/resample: {int(round(args.frac * len(all_pids)))}")

    meta = {
        "n_resamples": args.n_resamples,
        "frac": args.frac,
        "n_participants": int(len(all_pids)),
        "n_keep": int(round(args.frac * len(all_pids))),
        "cv_repeat_scale": args.cv_repeat_scale,
        "methods": list(METHODS),
        "seed": args.seed,
        "note": (
            "Participant-level subsample without replacement; "
            "Stage1 scoring + notable selection redone each resample. "
            "freq = n_selected / n_resamples (ineligible resamples count as not selected)."
        ),
    }

    if args.experiment in ("items", "both"):
        items = items_of(lab)
        print("Building item cache…")
        cache = build_item_cache(feat, lab, items)
        print(f"item cells: {len(cache)}")
        freq, _ = run_items_stability(
            cache,
            all_pids,
            n_resamples=args.n_resamples,
            frac=args.frac,
            top_k=args.top_k_items,
            auc_min=args.auc_min_items,
            jobs=args.jobs,
            seed=args.seed,
            cv_repeat_scale=args.cv_repeat_scale,
        )
        path = args.out / "items_selection_freq.csv"
        freq.to_csv(path, index=False)
        print(f"\nItems top 25 by freq:\n{freq.head(25).to_string(index=False)}")
        print(f"-> {path}")
        # merge with original Stage2 for context if present
        s2 = ROOT / "outputs/experiment/within_fold_items/stage2_perm.csv"
        if s2.exists():
            s2df = pd.read_csv(s2)
            m = freq.merge(
                s2df[["task", "role", "item", "auc", "p", "method"]],
                on=["task", "role", "item"],
                how="left",
                suffixes=("", "_stage2"),
            )
            m.to_csv(args.out / "items_freq_with_stage2.csv", index=False)
        meta["items"] = {
            "n_cells": len(cache),
            "top_k": args.top_k_items,
            "auc_min": args.auc_min_items,
            "n_freq_ge_0.5": int((freq.freq >= 0.5).sum()),
            "n_freq_ge_0.8": int((freq.freq >= 0.8).sum()),
        }

    if args.experiment in ("aggregates", "both"):
        print("Building aggregate cache…")
        cache = build_agg_cache(feat, lab)
        print(f"aggregate cells: {len(cache)}")
        freq = run_agg_stability(
            cache,
            all_pids,
            n_resamples=args.n_resamples,
            frac=args.frac,
            jobs=args.jobs,
            seed=args.seed + 17,
            cv_repeat_scale=args.cv_repeat_scale,
        )
        path = args.out / "aggregates_selection_freq.csv"
        freq.to_csv(path, index=False)
        print(f"\nAggregates top 25 by freq:\n{freq.head(25).to_string(index=False)}")
        print(f"-> {path}")
        s2 = ROOT / "outputs/experiment/within_fold_aggregates/stage2_perm.csv"
        if s2.exists():
            s2df = pd.read_csv(s2)
            m = freq.merge(
                s2df[["task", "role", "target", "mode", "auc", "r", "p", "method"]],
                on=["task", "role", "target", "mode"],
                how="left",
                suffixes=("", "_stage2"),
            )
            m.to_csv(args.out / "aggregates_freq_with_stage2.csv", index=False)
        meta["aggregates"] = {
            "n_cells": len(cache),
            "top_k_binary": AGG_TOP_K,
            "top_k_regression": AGG_TOP_K,
            "auc_min": AGG_AUC_MIN,
            "r_min": R_MIN,
            "n_freq_ge_0.5": int((freq.freq >= 0.5).sum()),
            "n_freq_ge_0.8": int((freq.freq >= 0.8).sum()),
        }

    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"meta -> {args.out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
