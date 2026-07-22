"""End-to-end per-participant / per-task feature extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from ados_ml.data.tasks import (
    TARGET_TASK_IDS,
    TASK_NAMES,
    Interval,
    assign_task_ids,
    load_task_intervals,
    total_duration,
)
from ados_ml.features.face import accumulate_face_frame, empty_face_buffer, finalize_face
from ados_ml.features.pose import (
    accumulate_pose_frame,
    empty_pose_buffer,
    finalize_pose,
)
from ados_ml.features.speech import extract_speech_features
from ados_ml.utils.ids import normalize_participant_id


def _pick_role(instances: list[dict] | None, role: str) -> dict | None:
    if not instances:
        return None
    for inst in instances:
        if inst.get("role") == role:
            return inst
    return None


def _scan_participant(
    participant_id: str,
    data_root: Path,
    *,
    tau_pose: float,
    tau_face: float,
) -> tuple[
    str,
    dict[int, list[Interval]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
    list[dict],
    list[float],
    float,
]:
    pid = normalize_participant_id(participant_id)
    folder = data_root / pid
    session_csv = folder / f"{pid}_tasks_multimodal_v2_session.csv"
    session_json = folder / f"{pid}_multimodal_session_v1.json"
    if not session_csv.exists():
        raise FileNotFoundError(session_csv)
    if not session_json.exists():
        raise FileNotFoundError(session_json)

    intervals_by_task = load_task_intervals(session_csv)
    for tid in TARGET_TASK_IDS:
        if not intervals_by_task.get(tid):
            raise ValueError(f"{pid}: missing intervals for task {tid}")

    with session_json.open(encoding="utf-8") as f:
        doc = json.load(f)

    sample_interval = float(doc.get("sample_interval_sec") or 0.25)
    pose_bufs = {tid: empty_pose_buffer() for tid in TARGET_TASK_IDS}
    face_bufs = {tid: empty_face_buffer() for tid in TARGET_TASK_IDS}

    for fr in doc.get("timeline") or []:
        t = float(fr["time_sec"])
        tids = assign_task_ids(t, intervals_by_task)
        if not tids:
            continue
        poses = fr.get("pose") or []
        faces = fr.get("face") or []
        child_p = _pick_role(poses, "child")
        ex_p = _pick_role(poses, "examiner")
        child_f = _pick_role(faces, "child")
        ex_f = _pick_role(faces, "examiner")
        for tid in tids:
            accumulate_pose_frame(
                pose_bufs, tid, child_p, ex_p, t, tau_pose=tau_pose
            )
            accumulate_face_frame(
                face_bufs, tid, child_f, ex_f, tau_face=tau_face
            )

    dist_samples: list[float] = []
    for tid in TARGET_TASK_IDS:
        for d, w in zip(pose_bufs[tid]["dyad_dist"], pose_bufs[tid]["dyad_w"]):
            if d is not None and np.isfinite(d) and w > 0:
                dist_samples.append(float(d))

    speech_segments = doc.get("speech_segments") or []
    return (
        pid,
        intervals_by_task,
        pose_bufs,
        face_bufs,
        speech_segments,
        dist_samples,
        sample_interval,
    )


def _finalize_participant(
    pid: str,
    intervals_by_task: dict[int, list[Interval]],
    pose_bufs: dict[int, dict[str, Any]],
    face_bufs: dict[int, dict[str, Any]],
    speech_segments: list[dict],
    *,
    close_threshold: float | None,
    tau_sp: float,
    embed_fn: Callable[[list[str]], np.ndarray] | None,
    sample_interval: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tid in TARGET_TASK_IDS:
        ivs = intervals_by_task[tid]
        dur = total_duration(ivs)
        pose_feats = finalize_pose(
            pose_bufs[tid],
            dur,
            close_threshold=close_threshold,
            sample_interval=sample_interval,
        )
        face_feats = finalize_face(
            face_bufs[tid], dur, sample_interval=sample_interval
        )
        speech_feats = extract_speech_features(
            speech_segments,
            ivs,
            dur,
            tau_sp=tau_sp,
            embed_fn=embed_fn,
        )
        emb = speech_feats.pop("child_emb_mean", None)
        row: dict[str, Any] = {
            "participant_id": pid,
            "task_id": tid,
            "task_name": TASK_NAMES[tid],
            "task_duration_sec": dur,
            "n_task_segments": len(ivs),
            **pose_feats,
            **face_feats,
            **speech_feats,
        }
        if emb is not None:
            for i, val in enumerate(emb.tolist()):
                row[f"child_emb_{i}"] = val
        rows.append(row)
    return rows


def extract_cohort(
    participant_ids: list[str],
    data_root: Path,
    *,
    tau_pose: float = 0.25,
    tau_face: float = 0.25,
    tau_sp: float = 0.3,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Scan each session once; set dyad_close_frac with cohort median distance."""
    scanned: list[tuple] = []
    all_dist: list[float] = []

    for pid in tqdm(participant_ids, desc="scan_sessions"):
        item = _scan_participant(
            pid, data_root, tau_pose=tau_pose, tau_face=tau_face
        )
        scanned.append(item)
        all_dist.extend(item[5])

    close_thr = float("nan")
    if all_dist:
        positive = [d for d in all_dist if d > 1e-6]
        # Many frames report distance==0; threshold from positive distances only.
        base = positive if positive else all_dist
        close_thr = float(np.median(base))
    thr = close_thr if np.isfinite(close_thr) else None

    rows: list[dict[str, Any]] = []
    for item in tqdm(scanned, desc="finalize_features"):
        pid, intervals, pose_bufs, face_bufs, speech, _, sample_interval = item
        part_rows = _finalize_participant(
            pid,
            intervals,
            pose_bufs,
            face_bufs,
            speech,
            close_threshold=thr,
            tau_sp=tau_sp,
            embed_fn=embed_fn,
            sample_interval=sample_interval,
        )
        for r in part_rows:
            r["dyad_close_threshold"] = close_thr
        rows.extend(part_rows)

    df = pd.DataFrame(rows)
    df["participant_id"] = df["participant_id"].astype(str)
    front = [
        "participant_id",
        "task_id",
        "task_name",
        "task_duration_sec",
        "n_task_segments",
        "dyad_close_threshold",
    ]
    cols = front + [c for c in df.columns if c not in front]
    return df[cols]
