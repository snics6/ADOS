#!/usr/bin/env python
"""Exp3: more sign-flips on locked Δ3. Sensitivity; does not lock a new family.

    ./venv/bin/python -u scripts/run_exp3_resolution.py
    ./venv/bin/python -u scripts/run_exp3_resolution.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.exp3_resolution import (  # noqa: E402
    N_RESOLVE,
    compare_resolution,
    render_readme,
)
from ados_ffm.exp3_transplant import tests  # noqa: E402
from ados_ffm.exp34_labels import N_LABEL_PERM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sides", type=Path, default=ROOT / "outputs/exp3/sides.csv")
    ap.add_argument("--main", type=Path, default=ROOT / "outputs/exp3/main_test.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/exp3_resolution")
    ap.add_argument("--n-perm", type=int, default=N_RESOLVE)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_perm = args.n_perm
    if args.smoke:
        n_perm = min(n_perm, 20)
        if args.out == ROOT / "outputs/exp3_resolution":
            args.out = ROOT / "outputs/_smoke/exp3_resolution"

    if not args.sides.exists():
        raise SystemExit(f"exp3 sides not found: {args.sides}")
    if not args.main.exists():
        raise SystemExit(f"exp3 main_test not found: {args.main}")

    sides = pd.read_csv(args.sides)
    orig_main = pd.read_csv(args.main)
    hi_main, _sec = tests(sides, n_perm=n_perm)
    compare = compare_resolution(
        orig_main, hi_main, n_perm_orig=N_LABEL_PERM, n_perm_hi=n_perm
    )

    args.out.mkdir(parents=True, exist_ok=True)
    compare.to_csv(args.out / "compare.csv", index=False)
    hi_main.to_csv(args.out / "main_test.csv", index=False)
    readme = render_readme(compare, n_perm_orig=N_LABEL_PERM, n_perm_hi=n_perm)
    (args.out / "README.md").write_text(readme, encoding="utf-8")
    meta = {
        "n_perm_orig": N_LABEL_PERM,
        "n_perm_hi": n_perm,
        "locked_main_unchanged": True,
        "same_deltas_and_seeds": True,
        "sides": str(args.sides),
        "main": str(args.main),
        "smoke": bool(args.smoke),
        "n_same_sig": int((compare["sig_orig"] == compare["sig_hi"]).sum()),
        "n_sig_orig": int(compare["sig_orig"].sum()),
        "n_sig_hi": int(compare["sig_hi"].sum()),
    }
    (args.out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(compare[["source", "target", "p_orig", "p_hi", "sig_orig", "sig_hi"]].to_string(index=False))
    print(f"wrote {args.out / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
