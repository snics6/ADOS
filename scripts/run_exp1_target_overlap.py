#!/usr/bin/env python
"""Exp1: same (task, source, feature) hitting more than one of SA/RRB/CSS.

    ./venv/bin/python -u scripts/run_exp1_target_overlap.py
    ./venv/bin/python -u scripts/run_exp1_target_overlap.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hits", type=Path, default=ROOT / "outputs/exp1/hits_main.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/exp1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke and args.out == ROOT / "outputs/exp1":
        args.out = ROOT / "outputs/_smoke/exp1"
        if args.hits == ROOT / "outputs/exp1/hits_main.csv":
            smoke_hits = ROOT / "outputs/_smoke/exp1/hits_main.csv"
            if smoke_hits.exists():
                args.hits = smoke_hits

    if not args.hits.exists():
        raise SystemExit(f"hits not found: {args.hits}")
    hits = pd.read_csv(args.hits)
    key = ["task", "task_ja", "source", "feature"]
    args.out.mkdir(parents=True, exist_ok=True)
    if hits.empty:
        empty = pd.DataFrame(
            columns=[
                *key,
                "n_targets",
                "targets",
                "on_SA",
                "on_RRB",
                "on_CSS",
                "rho_SA",
                "rho_RRB",
                "rho_CSS",
                "n",
            ]
        )
        empty.to_csv(args.out / "target_overlap_all.csv", index=False)
        empty.to_csv(args.out / "target_overlap_multi.csv", index=False)
        meta = {
            "n_raw_hits": 0,
            "n_unique_task_source_feature": 0,
            "n_multi_target": 0,
            "n_exactly_2": 0,
            "n_exactly_3": 0,
            "extra_hits_from_overlap": 0,
            "smoke": bool(args.smoke),
        }
        (args.out / "target_overlap_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    g = hits.groupby(key, dropna=False)
    rows = []
    for k, sub in g:
        targets = sorted(sub["target"].unique())
        rows.append(
            {
                "task": int(k[0]),
                "task_ja": k[1],
                "source": k[2],
                "feature": k[3],
                "n_targets": len(targets),
                "targets": "|".join(targets),
                "on_SA": "SA" in targets,
                "on_RRB": "RRB" in targets,
                "on_CSS": "CSS" in targets,
                "rho_SA": float(sub.loc[sub["target"] == "SA", "rho"].iloc[0])
                if "SA" in targets
                else float("nan"),
                "rho_RRB": float(sub.loc[sub["target"] == "RRB", "rho"].iloc[0])
                if "RRB" in targets
                else float("nan"),
                "rho_CSS": float(sub.loc[sub["target"] == "CSS", "rho"].iloc[0])
                if "CSS" in targets
                else float("nan"),
                "n": int(sub["n"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["n_targets", "task", "source", "feature"], ascending=[False, True, True, True]
    )
    multi = out[out["n_targets"] >= 2]
    out.to_csv(args.out / "target_overlap_all.csv", index=False)
    multi.to_csv(args.out / "target_overlap_multi.csv", index=False)
    n_raw = int(len(hits))
    n_unique = int(len(out))
    n_multi = int(len(multi))
    n_triple = int((out["n_targets"] == 3).sum())
    n_double = int((out["n_targets"] == 2).sum())
    extra = n_raw - n_unique
    meta = {
        "n_raw_hits": n_raw,
        "n_unique_task_source_feature": n_unique,
        "n_multi_target": n_multi,
        "n_exactly_2": n_double,
        "n_exactly_3": n_triple,
        "extra_hits_from_overlap": extra,
        "smoke": bool(args.smoke),
    }
    (args.out / "target_overlap_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"multi={n_multi}  -> {args.out / 'target_overlap_multi.csv'}")


if __name__ == "__main__":
    main()
