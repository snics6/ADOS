#!/usr/bin/env python
"""Extract per-task features from session JSON.

Entry point: data -> tables in outputs/features/ used by experiments 1-4.

Two extractors, same JSON, same formulas as the 2026-08 run:

1. Head / body / speech dynamics
   Code: src/ados_extract/dynamics/
   Out:  outputs/features/dynamics/features_task.csv

2. Time windows, replies, gestures, ASR text
   Code: src/ados_extract/windows.py
   Out:  outputs/features/windows/features_task.csv

    ./venv/bin/python -u scripts/extract_task_features.py
    ./venv/bin/python -u scripts/extract_task_features.py --dynamics-only
    ./venv/bin/python -u scripts/extract_task_features.py --windows-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_extract.dynamics import extract_cohort  # noqa: E402
from ados_extract.windows import extract_windows  # noqa: E402
from ados_ffm.data import (  # noqa: E402
    DEFAULT_COHORT,
    DEFAULT_FEATURES,
    DEFAULT_WINDOWS,
    TASKS,
    load_cohort,
)


def _dynamics(ids: list[str], data_root: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"dynamics  n={len(ids)}  tasks={list(TASKS)}  ->  {out_csv}", flush=True)
    t0 = time.time()
    task_df, sess_df = extract_cohort(ids, data_root, task_ids=TASKS)
    task_df.to_csv(out_csv, index=False)
    meta = {
        "n_participants": int(sess_df["participant_id"].nunique()) if not sess_df.empty else 0,
        "n_task_rows": int(len(task_df)),
        "n_columns": int(task_df.shape[1]),
        "tasks": list(TASKS),
        "elapsed_sec": round(time.time() - t0, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code": "src/ados_extract/dynamics/",
        "data_root": str(data_root),
    }
    (out_csv.parent / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"dynamics  rows={len(task_df)}  cols={task_df.shape[1]}  "
        f"{meta['elapsed_sec']:.1f}s",
        flush=True,
    )


def _windows(ids: list[str], data_root: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"windows   n={len(ids)}  ->  {out_csv}", flush=True)
    t0 = time.time()
    new = extract_windows(data_root, ids, TASKS)
    new.to_csv(out_csv, index=False)
    print(f"windows   rows={len(new)}  cols={new.shape[1]}  {time.time()-t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    ap.add_argument("--dynamics-out", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows-out", type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument("--dynamics-only", action="store_true")
    ap.add_argument("--windows-only", action="store_true")
    args = ap.parse_args()
    ids = load_cohort(args.cohort)
    do_dyn = not args.windows_only
    do_win = not args.dynamics_only
    if do_dyn:
        _dynamics(ids, args.data_root, args.dynamics_out)
    if do_win:
        _windows(ids, args.data_root, args.windows_out)


if __name__ == "__main__":
    main()
