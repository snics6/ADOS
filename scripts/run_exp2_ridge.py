#!/usr/bin/env python
"""Experiment 2: Ridge on Exp1 hits (k=1 or 2). Writes outputs/exp2/.

    ./venv/bin/python -u scripts/run_exp2_ridge.py
    ./venv/bin/python -u scripts/run_exp2_ridge.py --smoke
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
from ados_ffm.exp2_ridge import (  # noqa: E402
    N_PERM,
    attach_fdr_task_target,
    prepare_jobs,
    run_one,
)
from ados_ffm.ridge_cv import SOURCES  # noqa: E402
from ados_ffm.kinds import feature_kind  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/exp2")
    ap.add_argument("--hits", type=Path, default=ROOT / "outputs/exp1/hits_main.csv")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", type=str, default="")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    task_ids = TASKS
    n_perm = args.n_perm
    if args.smoke:
        n_perm = min(n_perm, 5)
        task_ids = (12, 4)
        if args.out == ROOT / "outputs/exp2":
            args.out = ROOT / "outputs/_smoke/exp2"
    if args.tasks:
        task_ids = tuple(int(x) for x in args.tasks.split(",") if x.strip())

    args.out.mkdir(parents=True, exist_ok=True)
    hits = pd.read_csv(args.hits)
    cohort = load_cohort()
    lab = load_labels()
    dyn = load_features(path=args.features)
    dyn = dyn[dyn["participant_id"].astype(str).isin(cohort)].copy()
    new = pd.read_csv(args.windows)
    new["participant_id"] = new["participant_id"].astype(str)

    jobs, skipped = prepare_jobs(hits, dyn, new, lab, cohort, TARGETS, task_ids, SOURCES)
    for j in jobs:
        print(
            f"ready  {j['task']} {j['task_ja']} {j['source']} {j['target']}  "
            f"k={j['k']} n={j['n']}  {j['cols']}",
            flush=True,
        )
    print(f"exp2  jobs={len(jobs)}  skipped={len(skipped)}  n_perm={n_perm}", flush=True)
    t0 = time.time()
    got = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_one)(cell, n_perm) for cell in jobs
    )
    df = pd.DataFrame(list(got)).sort_values(["task", "target", "source"])
    if not df.empty:
        df = attach_fdr_task_target(df)
    df.to_csv(args.out / "cells.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.out / "skipped.csv", index=False)

    feat_rows = []
    for cell in jobs:
        for i, c in enumerate(cell["cols"]):
            feat_rows.append(
                {
                    "task": cell["task"],
                    "source": cell["source"],
                    "target": cell["target"],
                    "rank": i + 1,
                    "feature": c,
                    "kind": feature_kind(c),
                }
            )
    pd.DataFrame(feat_rows).to_csv(args.out / "selected_features.csv", index=False)

    meta = {
        "n_perm": n_perm,
        "n_jobs_ran": len(jobs),
        "n_skipped": len(skipped),
        "k_rule": "1 if 1 Exp1 hit else 2",
        "bh": "task x score; m = sources that ran",
        "hits": str(args.hits),
        "features": str(args.features),
        "windows": str(args.windows),
        "smoke": bool(args.smoke),
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not df.empty:
        print(df[["task", "source", "target", "k", "n", "rho", "p", "q_fdr", "m"]].to_string(index=False), flush=True)
    print(f"done  {meta['seconds']:.1f}s  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
