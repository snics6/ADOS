#!/usr/bin/env python
"""Exp3 data claims only: (λ,r) positions, CI forest, duration, Q ranks, initiator.

Does not re-run the 21 pair tests. Reads outputs/hetero/*.csv and writes
the non-tautological tables and figures.

    ./venv/bin/python -u scripts/run_exp3_findings.py
    ./venv/bin/python -u scripts/run_exp3_findings.py --reextract-initiators
    ./venv/bin/python -u scripts/run_exp3_findings.py --skip-maxstat --skip-initiators
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.data import DEFAULT_FEATURES, DEFAULT_WINDOWS  # noqa: E402
from ados_ffm.exp3_findings import N_PERM_LAM, run  # noqa: E402
from ados_ffm.hetero import N_BOOT_PAIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/hetero")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument("--hits", type=Path, default=ROOT / "outputs/exp1/hits_main.csv")
    ap.add_argument("--n-boot-maxstat", type=int, default=N_BOOT_PAIR)
    ap.add_argument("--n-perm-lambda", type=int, default=N_PERM_LAM)
    ap.add_argument("--skip-initiators", action="store_true")
    ap.add_argument("--maxstat", action="store_true", help="optional; tests are not the headline")
    ap.add_argument("--reextract-initiators", action="store_true")
    args = ap.parse_args()
    run(
        args.out,
        features=args.features,
        windows=args.windows,
        hits=args.hits,
        n_boot_maxstat=args.n_boot_maxstat,
        n_perm_lambda=args.n_perm_lambda,
        skip_initiators=args.skip_initiators,
        skip_maxstat=not args.maxstat,
        reextract_initiators=args.reextract_initiators,
    )


if __name__ == "__main__":
    main()
