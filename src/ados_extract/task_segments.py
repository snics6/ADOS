"""Canonical manual task segments (data/task_segments.json only).

Do not derive intervals from session JSON, flow relabel, or any other source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_SEGMENTS = ROOT / "data" / "task_segments.json"

# Sequential task ids from manual segmentation (1–10).
CANONICAL_TASK_IDS: tuple[int, ...] = tuple(range(1, 11))

TASK_JA: dict[int, str] = {
    1: "構成課題",
    2: "ごっこあそび",
    3: "共同注意",
    4: "実演",
    5: "絵の説明",
    6: "本のストーリーの説明",
    7: "自由遊び",
    8: "誕生日",
    9: "おやつ",
    10: "ルーティン",
}

def parse_time_text(text: str) -> float:
    """Parse manual time labels: ``1分40秒``, ``3分``, ``15秒``, ``10:30``, ``1:03:50``."""
    s = str(text).strip()
    s = re.sub(r"[:;]+", ":", s)
    if not s:
        raise ValueError("empty time string")

    parts = re.split(r"[:;]", s)
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return float(int(parts[0]) * 60 + int(parts[1]))

    if "分" in s or "秒" in s:
        m2 = re.match(r"^\s*(\d+)分(?:(\d+)(?:秒)?)?\s*$", s)
        if m2:
            return float(int(m2.group(1)) * 60 + int(m2.group(2) or 0))
        m3 = re.match(r"^\s*(\d+)秒\s*$", s)
        if m3:
            return float(int(m3.group(1)))

    m4 = re.match(r"^\s*(\d+)分(\d+)\s*$", s)
    if m4:
        return float(int(m4.group(1)) * 60 + int(m4.group(2)))

    raise ValueError(f"unrecognized time format: {text!r}")


def load_task_segments_json(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads((path or DEFAULT_TASK_SEGMENTS).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("task_segments.json must be a JSON object keyed by participant id")
    return {str(k): list(v) for k, v in raw.items()}


def merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of half-open intervals, as a sorted list of disjoint spans.

    Touching or overlapping spans are joined; a genuine hole between two
    stretches is kept as a hole.
    """
    xs = sorted((float(a), float(b)) for a, b in spans if b > a)
    if not xs:
        return []
    out = [xs[0]]
    for a, b in xs[1:]:
        if a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def segments_for_participant(
    pid: str,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]] | None:
    """Return session-style task_segments for one participant, or None if missing.

    A task interrupted and resumed is annotated as several rows with the same
    ``task_id``. Those rows are kept as **separate stretches** (one entry each,
    after taking their union), never collapsed into a single ``min(t0)``..
    ``max(t1)`` span. The convex hull would swallow whatever other activity ran
    in between -- for one participant it spanned 53 minutes and contained seven
    other tasks whole -- so features computed "inside" the task would be
    measuring other tasks. Consumers must therefore treat a task interval as a
    set of disjoint spans, not as one contiguous stretch.
    """
    all_segs = load_task_segments_json(path)
    pid = str(pid)
    if pid not in all_segs:
        return None

    spans_by_task: dict[int, list[tuple[float, float]]] = {}
    meta_by_task: dict[int, dict[str, Any]] = {}
    for row in all_segs[pid]:
        t0s, t1s = str(row.get("t0", "")).strip(), str(row.get("t1", "")).strip()
        if not t0s or not t1s:
            continue
        tid = int(row["task_id"])
        t0 = parse_time_text(t0s)
        t1 = parse_time_text(t1s)
        if t1 <= t0:
            continue
        spans_by_task.setdefault(tid, []).append((t0, t1))
        meta_by_task.setdefault(
            tid,
            {
                "stage": row.get("stage"),
                "stage_ja": row.get("stage_ja") or TASK_JA.get(tid, str(tid)),
            },
        )

    out: list[dict[str, Any]] = []
    for tid in sorted(spans_by_task):
        spans = merge_spans(spans_by_task[tid])
        for i, (t0, t1) in enumerate(spans):
            out.append(
                {
                    "task_id": tid,
                    "session_start_sec": t0,
                    "session_end_sec": t1,
                    "stage": meta_by_task[tid]["stage"],
                    "stage_ja": meta_by_task[tid]["stage_ja"],
                    "relabel": "manual_canonical",
                    "stretch_index": i,
                    "n_stretches": len(spans),
                }
            )
    return out or None


def load_task_segments_map(
    participant_ids: list[str],
    *,
    path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    empty: list[str] = []
    for pid in participant_ids:
        segs = segments_for_participant(pid, path=path)
        if segs is None:
            if pid in load_task_segments_json(path):
                empty.append(pid)
            else:
                missing.append(pid)
            continue
        out[pid] = segs
    if missing:
        print(f"task_segments: no entry for {len(missing)} ids (first: {missing[:3]})")
    if empty:
        print(f"task_segments: empty/incomplete for {len(empty)} ids (first: {empty[:3]})")
    return out
