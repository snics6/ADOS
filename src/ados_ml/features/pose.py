"""Pose child / dyad features for one task window set."""

from __future__ import annotations

from typing import Any

import numpy as np

from ados_ml.features.confidence import pose_weight
from ados_ml.features.stats import (
    safe_div,
    theil_sen_slope,
    weighted_mean,
    weighted_std,
)


def _event_coverage(events: list[dict] | None, event_type: str) -> float:
    if not events:
        return 0.0
    covered = 0.0
    for ev in events:
        if ev.get("type") != event_type:
            continue
        start = float(ev.get("start", 0.0))
        end = float(ev.get("end", start))
        covered += max(0.0, end - start)
    return covered


def accumulate_pose_frame(
    buffers: dict[int, dict[str, Any]],
    tid: int,
    child: dict | None,
    examiner: dict | None,
    time_sec: float,
    *,
    tau_pose: float,
) -> None:
    buf = buffers[tid]
    if child is not None:
        w = pose_weight(child)
        feats = child.get("features") or {}
        buf["child_w"].append(w)
        for key in (
            "left_elbow_angle_deg",
            "right_elbow_angle_deg",
            "left_wrist_speed",
            "right_wrist_speed",
            "hand_to_face_dist",
            "torso_lean_deg",
            "head_drop",
            "joint_jitter",
        ):
            buf[f"child_{key}"].append(feats.get(key))
            buf[f"child_{key}_w"].append(w)
        buf["arm_extended_cov"] += _event_coverage(child.get("events"), "arm_extended")

    if child is not None and examiner is not None:
        wc = pose_weight(child)
        we = pose_weight(examiner)
        w_dy = min(wc, we)
        if w_dy >= tau_pose:
            feats = child.get("features") or {}
            # Prefer child copy; fall back to examiner if missing
            dist = feats.get("inter_person_distance")
            facing = feats.get("facing_angle_diff_deg")
            if dist is None:
                dist = (examiner.get("features") or {}).get("inter_person_distance")
            if facing is None:
                facing = (examiner.get("features") or {}).get("facing_angle_diff_deg")
            buf["dyad_w"].append(w_dy)
            buf["dyad_t"].append(time_sec)
            buf["dyad_dist"].append(dist)
            buf["dyad_facing"].append(facing)
            buf["dyad_valid_frames"] += 1


def empty_pose_buffer() -> dict[str, Any]:
    keys = [
        "left_elbow_angle_deg",
        "right_elbow_angle_deg",
        "left_wrist_speed",
        "right_wrist_speed",
        "hand_to_face_dist",
        "torso_lean_deg",
        "head_drop",
        "joint_jitter",
    ]
    buf: dict[str, Any] = {
        "child_w": [],
        "dyad_w": [],
        "dyad_t": [],
        "dyad_dist": [],
        "dyad_facing": [],
        "dyad_valid_frames": 0,
        "arm_extended_cov": 0.0,
    }
    for k in keys:
        buf[f"child_{k}"] = []
        buf[f"child_{k}_w"] = []
    return buf


def finalize_pose(
    buf: dict[str, Any],
    task_dur: float,
    *,
    close_threshold: float | None,
    sample_interval: float = 0.25,
) -> dict[str, float]:
    def wm(key: str) -> float:
        return weighted_mean(buf[f"child_{key}"], buf[f"child_{key}_w"])

    def ws(key: str) -> float:
        return weighted_std(buf[f"child_{key}"], buf[f"child_{key}_w"])

    left_spd = wm("left_wrist_speed")
    right_spd = wm("right_wrist_speed")
    speeds = [s for s in (left_spd, right_spd) if np.isfinite(s)]
    wrist_mean = float(np.mean(speeds)) if speeds else float("nan")

    dy_w = buf["dyad_w"]
    dist = buf["dyad_dist"]
    facing = buf["dyad_facing"]
    times = buf["dyad_t"]

    close_frac = float("nan")
    if close_threshold is not None and dy_w:
        flags = []
        weights = []
        for d, w in zip(dist, dy_w):
            if d is None or not np.isfinite(d):
                continue
            flags.append(1.0 if d < close_threshold else 0.0)
            weights.append(w)
        if weights:
            close_frac = weighted_mean(flags, weights)

    face_each = float("nan")
    if dy_w:
        flags = []
        weights = []
        for ang, w in zip(facing, dy_w):
            if ang is None or not np.isfinite(ang):
                continue
            flags.append(1.0 if abs(ang) < 45.0 else 0.0)
            weights.append(w)
        if weights:
            face_each = weighted_mean(flags, weights)

    valid_time = buf["dyad_valid_frames"] * sample_interval

    return {
        "child_left_elbow_angle_wmean": wm("left_elbow_angle_deg"),
        "child_left_elbow_angle_wstd": ws("left_elbow_angle_deg"),
        "child_right_elbow_angle_wmean": wm("right_elbow_angle_deg"),
        "child_right_elbow_angle_wstd": ws("right_elbow_angle_deg"),
        "child_left_wrist_speed_wmean": left_spd,
        "child_right_wrist_speed_wmean": right_spd,
        "child_wrist_speed_wmean": wrist_mean,
        "child_hand_to_face_dist_wmean": wm("hand_to_face_dist"),
        "child_hand_to_face_dist_wstd": ws("hand_to_face_dist"),
        "child_torso_lean_wmean": wm("torso_lean_deg"),
        "child_torso_lean_wstd": ws("torso_lean_deg"),
        "child_head_drop_wmean": wm("head_drop"),
        "child_joint_jitter_wmean": wm("joint_jitter"),
        "child_arm_extended_rate": safe_div(buf["arm_extended_cov"], task_dur),
        "dyad_distance_wmean": weighted_mean(dist, dy_w),
        "dyad_distance_wstd": weighted_std(dist, dy_w),
        "dyad_distance_slope": theil_sen_slope(times, [d if d is not None else np.nan for d in dist]),
        "dyad_close_frac": close_frac,
        "dyad_facing_angle_diff_wmean": weighted_mean(facing, dy_w),
        "dyad_facing_angle_diff_wstd": weighted_std(facing, dy_w),
        "dyad_face_each_other_frac": face_each,
        "dyad_valid_time_frac": safe_div(valid_time, task_dur),
    }
