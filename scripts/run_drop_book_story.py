#!/usr/bin/env python
"""Drop book-story pairs from Exp3 far-mean Δ3. Sensitivity; does not lock a new family.

    ./venv/bin/python -u scripts/run_drop_book_story.py
    ./venv/bin/python -u scripts/run_drop_book_story.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.drop_book_story import (  # noqa: E402
    BOOK_JA,
    BOOK_TASK,
    REL_CHANGE_LIMIT,
    book_context,
    render_readme,
    run_check,
    sides_counts,
)
from ados_ffm.exp34_labels import N_LABEL_PERM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sides", type=Path, default=ROOT / "outputs/exp3/sides.csv")
    ap.add_argument("--main", type=Path, default=ROOT / "outputs/exp3/main_test.csv")
    ap.add_argument("--exp1", type=Path, default=ROOT / "outputs/exp1/hits_main.csv")
    ap.add_argument("--exp2", type=Path, default=ROOT / "outputs/exp2/cells.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/drop_book_story")
    ap.add_argument("--n-perm", type=int, default=N_LABEL_PERM)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_perm = args.n_perm
    if args.smoke:
        n_perm = min(n_perm, 5)
        if args.out == ROOT / "outputs/drop_book_story":
            args.out = ROOT / "outputs/_smoke/drop_book_story"

    if not args.sides.exists():
        raise SystemExit(f"exp3 sides not found: {args.sides}")
    if not args.main.exists():
        raise SystemExit(f"exp3 main_test not found: {args.main}")

    sides = pd.read_csv(args.sides)
    orig_main = pd.read_csv(args.main)
    compare, dropped, keep_main = run_check(sides, orig_main, n_perm=n_perm)
    counts = sides_counts(sides)
    context = None
    if args.exp1.exists() and args.exp2.exists():
        context = book_context(pd.read_csv(args.exp1), pd.read_csv(args.exp2))

    args.out.mkdir(parents=True, exist_ok=True)
    compare.to_csv(args.out / "compare.csv", index=False)
    dropped.to_csv(args.out / "dropped_sides.csv", index=False)
    keep_main.to_csv(args.out / "main_keep.csv", index=False)
    readme = render_readme(
        compare, counts=counts, context=context, n_perm=n_perm, limit=REL_CHANGE_LIMIT
    )
    (args.out / "README.md").write_text(readme, encoding="utf-8")
    meta = {
        "book_task": BOOK_TASK,
        "book_ja": BOOK_JA,
        "rel_change_limit": REL_CHANGE_LIMIT,
        "n_perm": n_perm,
        "locked_main_unchanged": True,
        "sides": str(args.sides),
        "main": str(args.main),
        "smoke": bool(args.smoke),
        **counts,
        **(context or {}),
        "n_verdict_robust": int((compare["verdict"] == "頑健").sum()),
        "n_verdict_no_book": int((compare["verdict"] == "本は入っていない").sum()),
        "n_verdict_depends": int((compare["verdict"] == "一部はこの課題に依存").sum()),
    }
    (args.out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(compare[["source", "target", "n_far_orig", "n_far_keep", "verdict"]].to_string(index=False))
    print(f"wrote {args.out / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
