#!/usr/bin/env python
"""Experiment 2 (main): LOPO + Exp1-like Stage A on train, k=1, sign(ρ)·x.

Cells = Exp1 FDR hits (task × target × source). Feature pool = all catalog
features for that source. Outer y-permutation re-selects inside every LOPO fold.
BH within (task × score). Results: outputs/exp2/.

Auxiliary (Exp1 hits as-is, Ridge): scripts/run_exp2_ridge.py → outputs/exp2_ridge/

    ./venv/bin/python -u scripts/run_exp2.py
    ./venv/bin/python -u scripts/run_exp2.py --n-perm 200
    ./venv/bin/python -u scripts/run_exp2.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.data import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_WINDOWS,
    TASKS,
    TARGETS,
    load_cohort,
    load_features,
    load_labels,
)
from ados_ffm.exp2 import (  # noqa: E402
    N_BOOT,
    attach_fdr_task_target,
    load_thresholds,
    prepare_jobs,
    run_one,
)
from ados_ffm.ridge_cv import SOURCES  # noqa: E402
from ados_ffm.kinds import feature_kind  # noqa: E402

# Nested LOPO+select is heavy; 200 matches exploratory honesty setting.
DEFAULT_N_PERM = 200


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/exp2")
    ap.add_argument("--hits", type=Path, default=ROOT / "outputs/exp1/hits_main.csv")
    ap.add_argument("--thresholds", type=Path, default=ROOT / "outputs/exp1/thresholds.json")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", type=str, default="")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    task_ids = TASKS
    n_perm = args.n_perm
    if args.smoke:
        n_perm = min(n_perm, 5)
        task_ids = (7, 9)
        if args.out == ROOT / "outputs/exp2":
            args.out = ROOT / "outputs/_smoke/exp2"
    if args.tasks:
        task_ids = tuple(int(x) for x in args.tasks.split(",") if x.strip())

    args.out.mkdir(parents=True, exist_ok=True)
    hits = pd.read_csv(args.hits)
    thr = load_thresholds(args.thresholds)
    cohort = load_cohort()
    lab = load_labels()
    dyn = load_features(path=args.features)
    dyn = dyn[dyn["participant_id"].astype(str).isin(cohort)].copy()
    new = pd.read_csv(args.windows)
    new["participant_id"] = new["participant_id"].astype(str)

    jobs, skipped = prepare_jobs(
        hits, dyn, new, lab, cohort, TARGETS, task_ids, SOURCES, thresholds=thr
    )
    for j in jobs:
        print(
            f"ready  {j['task']} {j['task_ja']} {j['source']} {j['target']}  "
            f"n={j['n']} pool={len(j['cols_pool'])} tau={j['tau']:.3f}",
            flush=True,
        )
    print(
        f"exp2 (main)  jobs={len(jobs)}  skipped={len(skipped)}  "
        f"n_perm={n_perm} n_boot={args.n_boot}  method=lopo_stageA_signx",
        flush=True,
    )
    t0 = time.time()
    got = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_one)(cell, n_perm, n_boot=args.n_boot) for cell in jobs
    )
    df = pd.DataFrame(list(got)).sort_values(["task", "target", "source"])
    if not df.empty:
        df = attach_fdr_task_target(df)
    df.to_csv(args.out / "cells.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.out / "skipped.csv", index=False)

    feat_rows = []
    for row in df.itertuples(index=False):
        if getattr(row, "cols", ""):
            feat_rows.append(
                {
                    "task": row.task,
                    "source": row.source,
                    "target": row.target,
                    "rank": 1,
                    "feature": row.cols,
                    "kind": feature_kind(row.cols),
                    "top1_frac": getattr(row, "top1_frac", float("nan")),
                }
            )
    pd.DataFrame(feat_rows).to_csv(args.out / "selected_features.csv", index=False)

    n_pass = int(df["fdr_sig"].sum()) if not df.empty else 0
    meta = {
        "method": "lopo_stageA_signx",
        "k_fixed": 1,
        "selection": "train StageA S>=locked_tau then max|Spearman|; predict sign(train rho)*x",
        "pool": "all catalog features for cell source",
        "cells": "Exp1 FDR hit cells only",
        "cv": "LOPO",
        "n_perm": n_perm,
        "n_boot": args.n_boot,
        "n_jobs_ran": len(jobs),
        "n_skipped": len(skipped),
        "n_fdr": n_pass,
        "bh": "task x score; m = sources that ran",
        "thresholds": str(args.thresholds),
        "hits": str(args.hits),
        "features": str(args.features),
        "windows": str(args.windows),
        "smoke": bool(args.smoke),
        "seconds": round(time.time() - t0, 1),
        "aux_ridge": "outputs/exp2_ridge",
    }
    (args.out / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not df.empty:
        cols_show = [
            c
            for c in [
                "task",
                "source",
                "target",
                "n",
                "cols",
                "rho",
                "p",
                "q_fdr",
                "fdr_sig",
                "m",
            ]
            if c in df.columns
        ]
        print(df[cols_show].to_string(index=False), flush=True)
        print(
            f"FDR {n_pass}/{len(df)}  mean rho={df['rho'].mean():.3f}  "
            f"median rho={df['rho'].median():.3f}",
            flush=True,
        )
    print(f"done  {meta['seconds']:.1f}s  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
