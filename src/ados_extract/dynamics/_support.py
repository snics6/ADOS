"""Small helpers used only by the dynamics extractor.

Copied from the archived ``ados_seg`` package so this tree does not import
``old_files``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

ALL_TASK_IDS: tuple[int, ...] = tuple(range(1, 15))


def normalize_participant_id(raw: str | int | float) -> str:
    if isinstance(raw, float):
        if raw.is_integer():
            raw = int(raw)
        else:
            raise ValueError(f"Non-integer ID: {raw}")
    s = str(raw).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    s = s.replace("-", "_")
    if s.isdigit() and len(s) < 6:
        s = s.zfill(6)
    return s


def clip01(x: float | None) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def face_weight(face: dict) -> float:
    return clip01(face.get("det_score")) * clip01(face.get("role_confidence"))


def theil_sen_slope(
    times: Sequence[float], values: Sequence[float], *, max_points: int = 400
) -> float:
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(v)
    t = t[mask]
    v = v[mask]
    n = len(t)
    if n < 2:
        return float("nan")
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        t = t[idx]
        v = v[idx]
        n = len(t)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = t[j] - t[i]
            if abs(dt) < 1e-9:
                continue
            slopes.append((v[j] - v[i]) / dt)
    if not slopes:
        return float("nan")
    return float(np.median(slopes))


def _bbox_center(bbox: Sequence[float] | None) -> tuple[float, float] | None:
    if bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if not all(np.isfinite([x1, y1, x2, y2])):
        return None
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _center_dist(
    a: tuple[float, float] | None, b: tuple[float, float] | None
) -> float:
    if a is None or b is None:
        return float("inf")
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def is_role_swap(
    prev_child: dict,
    prev_examiner: dict,
    child: dict,
    examiner: dict,
) -> bool:
    pc = _bbox_center(prev_child.get("bbox"))
    pe = _bbox_center(prev_examiner.get("bbox"))
    cc = _bbox_center(child.get("bbox"))
    ce = _bbox_center(examiner.get("bbox"))
    if None in (pc, pe, cc, ce):
        return False
    same = _center_dist(cc, pc) + _center_dist(ce, pe)
    cross = _center_dist(cc, pe) + _center_dist(ce, pc)
    return cross + 1e-6 < same
