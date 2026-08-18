"""Run the dynamic feature families over a cohort.

Produces one row per participant x task and one row per participant. The
session row is not a convenience: the coupling measures need long stretches
of both people being tracked, and inside a single task the longest such
stretch is typically 60 to 85 s, which supports only a handful of windows.

Every row also carries how much data it was computed from. A synchrony
estimate from four windows and one from forty are not the same quantity,
and the screening step needs to be able to tell them apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from ados_extract.dynamics._support import ALL_TASK_IDS
from ados_extract.dynamics.conversation import conversation_features
from ados_extract.dynamics.crossmodal import crossmodal_features
from ados_extract.dynamics.kinematics import body_features, head_features
from ados_extract.dynamics.series import GRID_SEC, build_session_series
from ados_extract.dynamics.sync import sync_features
from ados_extract.dynamics._support import normalize_participant_id

KEY_COLS = ("participant_id", "task_id")
META_COLS = (
    "participant_id",
    "task_id",
    "n_frames",
    "duration_sec",
    "child_face_ok_frac",
    "examiner_face_ok_frac",
    "child_pose_ok_frac",
    "examiner_pose_ok_frac",
    "both_face_frac",
    "role_swap_frac",
    "child_scale_px",
    "examiner_scale_px",
    "angle_jump_frac",
)


def _window_features(ss, speech, mask, t0: float, t1: float) -> dict[str, float]:
    f: dict[str, float] = {}
    for rs in (ss.child, ss.examiner):
        f.update(head_features(rs, mask))
        f.update(body_features(rs, mask))
    f.update(sync_features(ss, mask))
    f.update(conversation_features(speech, t0, t1))
    f.update(crossmodal_features(ss, speech, mask))
    return f


def _meta(ss, mask) -> dict[str, float]:
    n = int(mask.sum())
    return {
        "n_frames": float(n),
        "duration_sec": float(n * GRID_SEC),
        "child_face_ok_frac": float((mask & ss.child.face_ok).sum() / max(n, 1)),
        "examiner_face_ok_frac": float((mask & ss.examiner.face_ok).sum() / max(n, 1)),
        "child_pose_ok_frac": float((mask & ss.child.pose_ok).sum() / max(n, 1)),
        "examiner_pose_ok_frac": float(
            (mask & ss.examiner.pose_ok).sum() / max(n, 1)
        ),
        "both_face_frac": float((mask & ss.both_face).sum() / max(n, 1)),
        "role_swap_frac": float((mask & ss.role_swap).sum() / max(n, 1)),
        "child_scale_px": ss.child.diagnostics.get("scale_px_median", np.nan),
        "examiner_scale_px": ss.examiner.diagnostics.get("scale_px_median", np.nan),
        "angle_jump_frac": float(
            np.nanmean(
                [
                    ss.child.diagnostics.get("angle_jump_frac", np.nan),
                    ss.examiner.diagnostics.get("angle_jump_frac", np.nan),
                ]
            )
        ),
    }


def _halves(mask: np.ndarray) -> list[np.ndarray]:
    """Split a mask into two equal-length halves of populated frames."""
    idx = np.flatnonzero(mask)
    if len(idx) < 8:
        return []
    cut = idx[len(idx) // 2]
    a, b = np.zeros_like(mask), np.zeros_like(mask)
    a[idx[idx < cut]] = True
    b[idx[idx >= cut]] = True
    return [a, b]


def extract_participant(
    pid: str,
    data_root: Path,
    *,
    task_ids: Iterable[int] = ALL_TASK_IDS,
    min_task_sec: float = 30.0,
    halves: bool = False,
    session_halves: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    norm = normalize_participant_id(pid)
    path = data_root / pid / f"{pid}_multimodal_session_v1.json"
    with path.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    ss = build_session_series(doc["timeline"], norm)
    speech = doc.get("speech_segments") or []
    segments = doc.get("task_segments") or []

    rows: list[dict[str, Any]] = []
    if session_halves:
        # Two session-level estimates built from the first and the second
        # half of every task. Splitting whole tasks instead would confound
        # the comparison with which tasks landed on which side.
        acc = [np.zeros(ss.n, bool), np.zeros(ss.n, bool)]
        for tid in task_ids:
            segs = [s for s in segments if int(s.get("task_id", -1)) == tid]
            if not segs:
                continue
            m = np.zeros(ss.n, bool)
            for s in segs:
                m |= (ss.t >= float(s["session_start_sec"])) & (
                    ss.t < float(s["session_end_sec"])
                )
            if m.sum() * GRID_SEC < min_task_sec:
                continue
            for i, hm in enumerate(_halves(m)):
                acc[i] |= hm
        for i, hm in enumerate(acc):
            if not hm.any():
                continue
            ht = ss.t[hm]
            row = {"participant_id": norm, "task_id": -1, "half": i}
            row.update(_meta(ss, hm))
            row.update(_window_features(ss, speech, hm, float(ht[0]), float(ht[-1])))
            rows.append(row)
        full = acc[0] | acc[1]
        sess: dict[str, Any] = {"participant_id": norm, "task_id": -1}
        if full.any():
            ft = ss.t[full]
            sess.update(_meta(ss, full))
            sess.update(
                _window_features(ss, speech, full, float(ft[0]), float(ft[-1]))
            )
        return rows, sess

    for tid in task_ids:
        segs = [s for s in segments if int(s.get("task_id", -1)) == tid]
        if not segs:
            continue
        mask = np.zeros(ss.n, bool)
        for s in segs:
            mask |= (ss.t >= float(s["session_start_sec"])) & (
                ss.t < float(s["session_end_sec"])
            )
        if mask.sum() * GRID_SEC < min_task_sec:
            continue
        t0 = min(float(s["session_start_sec"]) for s in segs)
        t1 = max(float(s["session_end_sec"]) for s in segs)
        if halves:
            # Two independent estimates of the same cell, for split-half
            # reliability. Splitting by time rather than by alternating
            # frames keeps each half a real stretch of interaction, which
            # the windowed and recurrence measures require.
            for h, hm in enumerate(_halves(mask)):
                ht = ss.t[hm]
                row = {"participant_id": norm, "task_id": tid, "half": h}
                row.update(_meta(ss, hm))
                row.update(
                    _window_features(ss, speech, hm, float(ht[0]), float(ht[-1]))
                )
                rows.append(row)
            continue
        row: dict[str, Any] = {"participant_id": norm, "task_id": tid}
        row.update(_meta(ss, mask))
        row.update(_window_features(ss, speech, mask, t0, t1))
        rows.append(row)

    # Session row: the union of the identified tasks, not the raw recording,
    # so waiting and setup time stays out.
    smask = np.zeros(ss.n, bool)
    for s in segments:
        smask |= (ss.t >= float(s["session_start_sec"])) & (
            ss.t < float(s["session_end_sec"])
        )
    sess: dict[str, Any] = {"participant_id": norm, "task_id": -1}
    if smask.any():
        t0 = min(float(s["session_start_sec"]) for s in segments)
        t1 = max(float(s["session_end_sec"]) for s in segments)
        sess.update(_meta(ss, smask))
        sess.update(_window_features(ss, speech, smask, t0, t1))
    return rows, sess


def extract_cohort(
    participant_ids: list[str],
    data_root: Path,
    *,
    task_ids: Iterable[int] = ALL_TASK_IDS,
    min_task_sec: float = 30.0,
    halves: bool = False,
    session_halves: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows: list[dict[str, Any]] = []
    sess_rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for pid in tqdm(participant_ids, desc="dynamics"):
        try:
            rows, sess = extract_participant(
                pid,
                data_root,
                task_ids=task_ids,
                min_task_sec=min_task_sec,
                halves=halves,
                session_halves=session_halves,
            )
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{pid}: {exc}")
            continue
        task_rows.extend(rows)
        sess_rows.append(sess)
    if failed:
        print(f"skipped {len(failed)}: " + "; ".join(failed[:5]))

    task_df = pd.DataFrame(task_rows)
    sess_df = pd.DataFrame(sess_rows)
    for df in (task_df, sess_df):
        if not df.empty:
            df["participant_id"] = df["participant_id"].astype(str)
    return task_df, sess_df


def read_dynamics(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"participant_id": str})
