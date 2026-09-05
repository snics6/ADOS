#!/usr/bin/env python
"""Experiment 1: per-feature stability screen, then permutation on candidates.

Thresholds are written before real stabilities (no peeking).

    ./venv/bin/python scripts/run_exp1_univariate.py
    ./venv/bin/python scripts/run_exp1_univariate.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.data import (  # noqa: E402
    FEATURE_COLS,
    TASK_JA,
    TASKS,
    TARGETS,
    DEFAULT_FEATURES,
    DEFAULT_WINDOWS,
    load_cohort,
    load_features,
    load_labels,
)
from ados_ffm.exp1_univariate import (  # noqa: E402
    MIN_N,
    N_BOOT,
    N_NULL,
    N_PERM,
    SEED_NULL,
    SEED_PERM,
    SEED_REAL,
    THRESH_Q,
    attach_fdr,
    catalog_cols,
    feature_payloads,
    merge_task_catalog,
    null_stabilities_one_target,
    perm_one,
    real_stabilities,
    stable_int,
    y_map_for_task,
)


def _tgt_index(name: str) -> int:
    for i, t in enumerate(TARGETS):
        if t["name"] == name:
            return i
    raise KeyError(name)


def _prepare_task(
    dyn: pd.DataFrame,
    new: pd.DataFrame,
    lab: pd.DataFrame,
    cohort: list[str],
    task_id: int,
    target: dict,
) -> tuple[list[str], dict[str, float], list[dict]] | None:
    frame = merge_task_catalog(dyn, new, task_id, cohort)
    if frame.empty:
        return None
    ymap = y_map_for_task(frame, lab, target)
    if len(ymap) < MIN_N:
        return None
    cols = catalog_cols(list(frame.columns))
    payloads = feature_payloads(frame, ymap, cols)
    if not payloads:
        return None
    pids_task = [p for p in frame["participant_id"].astype(str).tolist() if p in ymap]
    pids_task = list(dict.fromkeys(pids_task))
    return pids_task, ymap, payloads


def _null_job(task_id: int, target: dict, packed: tuple, n_null: int, n_boot: int):
    pids_task, ymap, payloads = packed
    seed = SEED_NULL + int(task_id) * 30 + _tgt_index(target["name"]) * 3
    rows = null_stabilities_one_target(
        payloads,
        pids_task,
        ymap,
        n_null=n_null,
        n_boot=n_boot,
        seed=seed,
    )
    for r in rows:
        r["task"] = int(task_id)
        r["task_ja"] = TASK_JA[task_id]
        r["target"] = target["name"]
    return rows


def _real_job(task_id: int, target: dict, packed: tuple, n_boot: int):
    _pids, _ymap, payloads = packed
    seed = SEED_REAL + int(task_id) * 30 + _tgt_index(target["name"]) * 3
    rows = real_stabilities(payloads, n_boot=n_boot, seed=seed)
    for r in rows:
        r["task"] = int(task_id)
        r["task_ja"] = TASK_JA[task_id]
        r["target"] = target["name"]
    return rows


def _perm_job(row: dict, packed_by: dict, n_perm: int):
    key = (int(row["task"]), row["target"])
    _pids, _ymap, payloads = packed_by[key]
    pl = next(p for p in payloads if p["feature"] == row["feature"])
    seed = SEED_PERM + int(row["task"]) * 300 + _tgt_index(row["target"]) * 40
    seed += stable_int(str(row["feature"]))
    out = perm_one(pl["x"], pl["y"], n_perm=n_perm, seed=seed)
    return {**row, **out, "n": pl["n"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/exp1")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument("--tasks", type=str, default="")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    if args.smoke:
        args.n_boot = min(args.n_boot, 20)
        args.n_null = min(args.n_null, 5)
        args.n_perm = min(args.n_perm, 50)
        task_ids = (7, 9)
        if args.out == ROOT / "outputs/exp1":
            args.out = ROOT / "outputs/_smoke/exp1"
    elif args.tasks:
        task_ids = tuple(int(x) for x in args.tasks.split(",") if x.strip())
    else:
        task_ids = TASKS

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.features.exists():
        raise SystemExit(f"dynamics table not found: {args.features}\nRun scripts/extract_task_features.py first.")
    if not args.windows.exists():
        raise SystemExit(f"windows table not found: {args.windows}\nRun scripts/extract_task_features.py first.")

    cohort = load_cohort()
    lab = load_labels()
    print(f"exp1  tasks={task_ids}  cohort={len(cohort)}", flush=True)
    t0 = time.time()

    dyn = load_features(path=args.features)
    dyn = dyn[dyn["participant_id"].astype(str).isin(cohort)].copy()
    new = pd.read_csv(args.windows)
    new["participant_id"] = new["participant_id"].astype(str)
    print(f"features {args.features}  rows={len(dyn)}", flush=True)
    print(f"windows  {args.windows}  rows={len(new)}", flush=True)

    packed: dict[tuple[int, str], tuple] = {}
    for tid in task_ids:
        for tgt in TARGETS:
            got = _prepare_task(dyn, new, lab, cohort, tid, tgt)
            if got is None:
                print(f"skip empty {tid} {tgt['name']}", flush=True)
                continue
            packed[(tid, tgt["name"])] = got
            print(
                f"ready  {tid} {TASK_JA[tid]} {tgt['name']}  "
                f"people={len(got[0])}  feats={len(got[2])}",
                flush=True,
            )

    jobs = [
        delayed(_null_job)(tid, tgt, packed[(tid, tgt["name"])], args.n_null, args.n_boot)
        for tid in task_ids
        for tgt in TARGETS
        if (tid, tgt["name"]) in packed
    ]
    print(f"stage threshold  jobs={len(jobs)}  n_null={args.n_null}  n_boot={args.n_boot}", flush=True)
    null_chunks = Parallel(n_jobs=args.n_jobs, verbose=5)(jobs)
    null_df = pd.DataFrame([r for chunk in null_chunks for r in chunk])
    null_df.to_csv(args.out / "null_stability.csv", index=False)

    thresholds: dict[str, dict[str, float]] = {t["name"]: {} for t in TARGETS}
    n_null_used: dict[str, dict[str, int]] = {t["name"]: {} for t in TARGETS}
    for (tid, tname), g in null_df.groupby(["task", "target"]):
        vals = g["stability"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        thr = float(np.quantile(vals, THRESH_Q)) if vals.size else float("nan")
        thresholds[str(tname)][str(int(tid))] = thr
        n_null_used[str(tname)][str(int(tid))] = int(vals.size)

    lock = {
        "quantile": THRESH_Q,
        "n_null": args.n_null,
        "n_boot": args.n_boot,
        "min_n": MIN_N,
        "catalog": "frozen34 + win_/rsp_/ges_/txt_ (no keywords)",
        "thresholds": thresholds,
        "n_null_values": n_null_used,
        "note": "Thresholds locked before real stabilities were written.",
    }
    lock_path = args.out / "thresholds.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"LOCKED {lock_path}", flush=True)
    for tname, by_task in thresholds.items():
        bits = " ".join(f"{tid}:{thr:.3f}" for tid, thr in sorted(by_task.items(), key=lambda x: int(x[0])))
        print(f"  {tname}  {bits}", flush=True)

    real_jobs = [
        delayed(_real_job)(tid, tgt, packed[(tid, tgt["name"])], args.n_boot)
        for tid in task_ids
        for tgt in TARGETS
        if (tid, tgt["name"]) in packed
    ]
    print(f"stage A  jobs={len(real_jobs)}", flush=True)
    real_chunks = Parallel(n_jobs=args.n_jobs, verbose=5)(real_jobs)
    stage_a = pd.DataFrame([r for chunk in real_chunks for r in chunk])
    stage_a["threshold"] = [
        thresholds[r["target"]][str(int(r["task"]))] for _, r in stage_a.iterrows()
    ]
    stage_a["candidate"] = stage_a["stability"] >= stage_a["threshold"]
    stage_a.to_csv(args.out / "stage_a.csv", index=False)
    print(
        f"stage A  rows={len(stage_a)}  candidates={int(stage_a['candidate'].sum())}",
        flush=True,
    )

    cand = stage_a[stage_a["candidate"]].copy()
    cand.to_csv(args.out / "candidates.csv", index=False)
    if cand.empty:
        print("no candidates; skip stage B", flush=True)
        meta = _meta(args, task_ids, t0, 0)
        (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    print(f"stage B  candidates={len(cand)}  n_perm={args.n_perm}", flush=True)
    perm_rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_perm_job)(r.to_dict(), packed, args.n_perm) for _, r in cand.iterrows()
    )
    stage_b = pd.DataFrame(perm_rows)
    keep = [
        "task",
        "task_ja",
        "target",
        "source",
        "family",
        "feature",
        "asr_risky",
        "n",
        "stability",
        "threshold",
        "rho",
        "p",
        "null_med",
        "null_p95",
    ]
    stage_b = stage_b[[c for c in keep if c in stage_b.columns]]
    stage_b = attach_fdr(stage_b)
    stage_b = stage_b.sort_values(["q_family", "p", "task", "target"])
    stage_b.to_csv(args.out / "stage_b.csv", index=False)
    main_hits = stage_b[stage_b["fdr_sig"]].copy()
    main_hits.to_csv(args.out / "hits_main.csv", index=False)
    print(
        f"stage B  candidates={len(stage_b)}  FDR hits={len(main_hits)}  "
        f"(family=task×target)",
        flush=True,
    )
    if len(main_hits):
        print(
            main_hits[["task_ja", "target", "source", "feature", "n", "rho", "p", "q_fdr"]].to_string(
                index=False
            ),
            flush=True,
        )

    meta = _meta(args, task_ids, t0, int(len(cand)))
    meta["n_stage_a"] = int(len(stage_a))
    meta["n_candidates"] = int(len(cand))
    meta["n_main_fdr_hits"] = int(len(main_hits))
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done  {time.time()-t0:.1f}s  -> {args.out}", flush=True)


def _meta(args, task_ids, t0, n_cand: int) -> dict:
    return {
        "tasks": list(task_ids),
        "targets": [t["name"] for t in TARGETS],
        "n_boot": args.n_boot,
        "n_null": args.n_null,
        "n_perm": args.n_perm,
        "threshold_quantile": THRESH_Q,
        "min_n": MIN_N,
        "metric": "spearman",
        "stage_a": "half_split_same_sign_abs_rho_ge_0.20",
        "perm": "two_sided_abs_rho",
        "fdr_family": "task_x_target",
        "same_sample_confirmation": True,
        "catalog_n_frozen": len(FEATURE_COLS),
        "features": str(args.features),
        "windows": str(args.windows),
        "seconds": time.time() - t0,
        "n_candidates": n_cand,
        "task_segments": "data/task_segments.json",
        "task_segments_source": "manual_canonical",
    }


if __name__ == "__main__":
    main()
