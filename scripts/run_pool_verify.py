#!/usr/bin/env python
"""Procedure 3: Eq. 3 prediction vs measured ρ_z on the 10 sign-flip pairs.

Not a test. Intersection sample, rank-then-pool (P1), percentile CIs.
Ranks are recomputed inside each bootstrap replicate.

    ./venv/bin/python -u scripts/run_pool_verify.py
    ./venv/bin/python -u scripts/run_pool_verify.py --self-check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.data import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_WINDOWS,
    TASKS,
    load_cohort,
    load_features,
    load_labels,
)
from ados_ffm.hetero import (  # noqa: E402
    N_BOOT_VERIFY,
    build_panel,
    plot_eq3_verify,
    pool_verify_summary,
    self_check,
    verify_signflip_table,
)

FLIP_DEFAULT = ROOT / "outputs/hetero/signflip_10.csv"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return v if np.isfinite(v) else None
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_pool_verify(panel, flips: pd.DataFrame, out: Path, n_boot: int) -> tuple[pd.DataFrame, dict]:
    table = verify_signflip_table(panel, flips, n_boot=n_boot)
    summary = pool_verify_summary(table)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "pool_verify_10.csv", index=False)
    plot_eq3_verify(table, out / "fig_ci_forest.png")
    return table, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/hetero")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument("--flips", type=Path, default=FLIP_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_VERIFY)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    failed = self_check()
    if failed:
        print("self-check FAILED:", *failed, sep="\n  ", flush=True)
        raise SystemExit(1)
    print("self-check ok", flush=True)
    if args.self_check and not args.smoke:
        return

    if args.smoke:
        args.n_boot = min(args.n_boot, 80)
        if args.out == ROOT / "outputs/hetero":
            args.out = ROOT / "outputs/_smoke/hetero"

    if not args.flips.exists():
        raise SystemExit(f"sign-flip table not found: {args.flips}")

    t0 = time.perf_counter()
    cohort = load_cohort()
    dyn = load_features(args.features)
    win = pd.read_csv(args.windows)
    win["participant_id"] = win["participant_id"].astype(str)
    lab = load_labels()
    panel = build_panel(dyn, win, lab, cohort, TASKS)
    flips = pd.read_csv(args.flips)
    print(f"verify  {len(flips)} sign-flip pairs  B={args.n_boot}", flush=True)
    table, summary = write_pool_verify(panel, flips, args.out, args.n_boot)
    if int(summary.get("n_match_cap") or 0) != int(summary["n_pairs"]):
        print(
            f"WARNING  intersection ρ mismatch vs signflip_10  "
            f"{summary.get('n_match_cap')}/{summary['n_pairs']}",
            flush=True,
        )
    print(
        f"eq3  median |pred-obs|={summary['median_abs_err']:.5f}  "
        f"max={summary['max_abs_err']:.5f}  "
        f"corr={summary['corr_pred_obs']:.3f}  "
        f"ρz CI covers 0: {summary['n_rho_z_covers_0']}/{summary['n_pairs']}  "
        f"ρ+ CI excludes 0: {summary['n_rho_pos_excludes_0']}/{summary['n_pairs']}",
        flush=True,
    )
    meta_path = args.out / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["pool_verify"] = {
        "role": "eq3_prediction_vs_measured_on_signflip_10",
        "not_a_test": True,
        "definition": "P1_rank_then_pool_within_intersection",
        "ci": "subject_bootstrap_percentile_rerank_inside",
        "n_boot": int(args.n_boot),
        **summary,
        "seconds": float(time.perf_counter() - t0),
    }
    meta_path.write_text(json.dumps(_jsonable(meta), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out / 'pool_verify_10.csv'}  {args.out / 'fig_ci_forest.png'}", flush=True)


if __name__ == "__main__":
    main()
