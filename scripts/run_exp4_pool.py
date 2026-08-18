#!/usr/bin/env python
"""Experiment 4: average own rank-1 across two tasks vs this task alone.

    ./venv/bin/python -u scripts/run_exp4_pool.py --labels-only
    ./venv/bin/python -u scripts/run_exp4_pool.py
    ./venv/bin/python -u scripts/run_exp4_pool.py --smoke
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
    TARGETS,
    load_cohort,
    load_features,
    load_labels,
)
from ados_ffm.exp1_univariate import merge_task_catalog  # noqa: E402
from ados_ffm.exp4_pool import (  # noqa: E402
    expand_sides,
    load_exp2_fdr,
    lock_labels,
    run_side,
    tests,
)
from ados_ffm.exp34_labels import N_LABEL_PERM  # noqa: E402

TGT = {t["name"]: t for t in TARGETS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/exp4")
    ap.add_argument("--exp2", type=Path, default=ROOT / "outputs/exp2/cells.csv")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument("--n-perm", type=int, default=N_LABEL_PERM)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    n_perm = args.n_perm
    if args.smoke:
        n_perm = min(n_perm, 5)
        if args.out == ROOT / "outputs/exp4":
            args.out = ROOT / "outputs/_smoke/exp4"

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.exp2.exists():
        raise SystemExit(f"exp2 cells not found: {args.exp2}")
    exp2 = load_exp2_fdr(args.exp2)
    labels = lock_labels(exp2)
    labels.to_csv(args.out / "near_far_labels.csv", index=False)
    print(f"wrote labels  pairs={len(labels)}  -> {args.out / 'near_far_labels.csv'}", flush=True)
    if len(labels):
        print(labels["label"].value_counts().to_string(), flush=True)
    if args.labels_only:
        meta = {
            "labels_only": True,
            "n_pairs_labeled": int(len(labels)),
            "n_near": int((labels["label"] == "近い").sum()) if len(labels) else 0,
            "n_far": int((labels["label"] == "遠い").sum()) if len(labels) else 0,
            "exp2": str(args.exp2),
        }
        (args.out / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    sides = expand_sides(labels)
    if args.smoke:
        sides = sides.head(12).copy()
    cohort = load_cohort()
    lab = load_labels()
    dyn = load_features(path=args.features)
    dyn = dyn[dyn["participant_id"].astype(str).isin(cohort)].copy()
    new = pd.read_csv(args.windows)
    new["participant_id"] = new["participant_id"].astype(str)
    task_ids = sorted(set(sides["task_onto"].astype(int)) | set(sides["task_from"].astype(int)))
    tabs = {tid: merge_task_catalog(dyn, new, tid, cohort) for tid in task_ids}

    jobs = sides.to_dict("records")
    print(f"exp4  sides={len(jobs)}  n_perm={n_perm}", flush=True)
    t0 = time.time()
    got = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_side)(row, tabs, lab, TGT[row["target"]]) for row in jobs
    )
    rows = [r for r in got if r is not None]
    skipped = pd.DataFrame([row for row, r in zip(jobs, got) if r is None])
    out_sides = pd.DataFrame(rows)
    if not out_sides.empty:
        out_sides = out_sides.sort_values(
            ["source", "target", "task_onto", "task_from"]
        ).reset_index(drop=True)
    out_sides.to_csv(args.out / "sides.csv", index=False)
    if not skipped.empty:
        skipped.to_csv(args.out / "skipped.csv", index=False)
    main, sec = (
        tests(out_sides, n_perm=n_perm)
        if not out_sides.empty
        else (pd.DataFrame(), pd.DataFrame())
    )
    main.to_csv(args.out / "main_test.csv", index=False)
    sec.to_csv(args.out / "secondary_test.csv", index=False)
    meta = {
        "n_perm": n_perm,
        "n_pairs_labeled": int(len(labels)),
        "n_sides_planned": int(len(jobs)),
        "n_sides_ran": int(len(out_sides)),
        "n_sides_skipped_n": int(len(skipped)),
        "bh_main": "source x score; m = combos with at least one far side",
        "bh_secondary": "source x score; m = combos with both near and far; separate family",
        "exp2": str(args.exp2),
        "features": str(args.features),
        "windows": str(args.windows),
        "smoke": bool(args.smoke),
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not main.empty:
        print(main.to_string(index=False), flush=True)
    print(f"done  {meta['seconds']:.1f}s  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
