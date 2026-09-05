#!/usr/bin/env python
"""Extract per-task features from session JSON.

Entry point: data -> tables in outputs/features/ used by experiments 1-4.

Task intervals: **only** ``data/task_segments.json`` (manual canonical).

    ./venv/bin/python -u scripts/extract_task_features.py
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
from ados_extract.task_segments import (  # noqa: E402
    DEFAULT_TASK_SEGMENTS,
    load_task_segments_map,
)
from ados_extract.windows import extract_windows  # noqa: E402
from ados_ffm.data import (  # noqa: E402
    DEFAULT_COHORT,
    DEFAULT_FEATURES,
    DEFAULT_WINDOWS,
    TASKS,
    load_cohort,
)


def _dynamics(
    ids: list[str],
    data_root: Path,
    out_csv: Path,
    *,
    task_ids: tuple[int, ...],
    segments_by_pid: dict[str, list[dict]],
    segments_path: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"dynamics  n={len(ids)}  tasks={list(task_ids)}  "
        f"segments={segments_path.name}  ->  {out_csv}",
        flush=True,
    )
    t0 = time.time()
    task_df, sess_df = extract_cohort(
        ids,
        data_root,
        task_ids=task_ids,
        min_task_sec=20.0,
        task_segments_by_pid=segments_by_pid,
    )
    task_df.to_csv(out_csv, index=False)
    meta = {
        "n_participants": int(sess_df["participant_id"].nunique())
        if not sess_df.empty
        else 0,
        "n_task_rows": int(len(task_df)),
        "n_columns": int(task_df.shape[1]),
        "tasks": list(task_ids),
        "elapsed_sec": round(time.time() - t0, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code": "src/ados_extract/dynamics/ + src/ados_extract/task_segments.py",
        "data_root": str(data_root),
        "task_segments": str(segments_path),
        "task_segments_source": "manual_canonical",
    }
    (out_csv.parent / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"dynamics  rows={len(task_df)}  cols={task_df.shape[1]}  "
        f"{meta['elapsed_sec']:.1f}s",
        flush=True,
    )


def _windows(
    ids: list[str],
    data_root: Path,
    out_csv: Path,
    *,
    task_ids: tuple[int, ...],
    segments_by_pid: dict[str, list[dict]],
    segments_path: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"windows   n={len(ids)}  segments={segments_path.name}  ->  {out_csv}",
        flush=True,
    )
    t0 = time.time()
    new = extract_windows(
        data_root, ids, task_ids, task_segments_by_pid=segments_by_pid
    )
    new.to_csv(out_csv, index=False)
    print(
        f"windows   rows={len(new)}  cols={new.shape[1]}  {time.time()-t0:.1f}s",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    ap.add_argument(
        "--task-segments",
        type=Path,
        default=DEFAULT_TASK_SEGMENTS,
        help="Manual canonical task intervals (default: data/task_segments.json)",
    )
    ap.add_argument("--dynamics-out", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows-out", type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument("--dynamics-only", action="store_true")
    ap.add_argument("--windows-only", action="store_true")
    args = ap.parse_args()
    ids = load_cohort(args.cohort)
    task_ids = tuple(TASKS)
    segments_by_pid = load_task_segments_map(ids, path=args.task_segments)
    print(
        f"task_segments loaded for {len(segments_by_pid)}/{len(ids)} people "
        f"from {args.task_segments}",
        flush=True,
    )
    if len(segments_by_pid) < len(ids):
        missing = [pid for pid in ids if pid not in segments_by_pid]
        raise SystemExit(
            f"missing or empty task segments for {len(missing)} ids: {missing[:5]}"
        )

    do_dyn = not args.windows_only
    do_win = not args.dynamics_only
    if do_dyn:
        _dynamics(
            ids,
            args.data_root,
            args.dynamics_out,
            task_ids=task_ids,
            segments_by_pid=segments_by_pid,
            segments_path=args.task_segments,
        )
    if do_win:
        _windows(
            ids,
            args.data_root,
            args.windows_out,
            task_ids=task_ids,
            segments_by_pid=segments_by_pid,
            segments_path=args.task_segments,
        )


if __name__ == "__main__":
    main()
