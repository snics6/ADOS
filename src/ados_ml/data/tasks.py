"""Task interval loading and merging."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


TARGET_TASK_IDS = (3, 4, 11, 12)

TASK_NAMES = {
    3: "pretend_play",
    4: "interactive_play",
    11: "birthday_party",
    12: "snack",
}


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    def overlap(self, a: float, b: float) -> float:
        return max(0.0, min(self.end, b) - max(self.start, a))


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: (x.start, x.end))
    out = [ordered[0]]
    for iv in ordered[1:]:
        last = out[-1]
        if iv.start <= last.end:
            out[-1] = Interval(last.start, max(last.end, iv.end))
        else:
            out.append(iv)
    return out


def load_task_intervals(
    session_csv: Path,
    task_ids: tuple[int, ...] = TARGET_TASK_IDS,
) -> dict[int, list[Interval]]:
    """Load and merge session-timeline intervals per task_id."""
    by_id: dict[int, list[Interval]] = {tid: [] for tid in task_ids}
    with session_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = int(row["task_id"])
            if tid not in by_id:
                continue
            start = float(row["session_start_sec"])
            end = float(row["session_end_sec"])
            if end > start:
                by_id[tid].append(Interval(start, end))
    return {tid: merge_intervals(ivs) for tid, ivs in by_id.items()}


def total_duration(intervals: list[Interval]) -> float:
    return float(sum(iv.duration for iv in intervals))


def assign_task_ids(t: float, intervals_by_task: dict[int, list[Interval]]) -> list[int]:
    hits = []
    for tid, ivs in intervals_by_task.items():
        if any(iv.contains(t) for iv in ivs):
            hits.append(tid)
    return hits
