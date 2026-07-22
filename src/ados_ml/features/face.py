"""Face child / dyad features for one task window set."""

from __future__ import annotations

from typing import Any

import numpy as np

from ados_ml.features.confidence import face_weight
from ados_ml.features.stats import safe_div, weighted_mean, weighted_quantile, weighted_std


def empty_face_buffer() -> dict[str, Any]:
    return {
        "c_w": [],
        "c_mar": [],
        "c_yaw": [],
        "c_pitch": [],
        "c_roll": [],
        "c_yaw_abs": [],
        "d_w": [],
        "d_yaw_diff": [],
        "d_mutual": [],
        "d_mar_c": [],
        "d_mar_e": [],
        "dyad_valid_frames": 0,
    }


def accumulate_face_frame(
    buffers: dict[int, dict[str, Any]],
    tid: int,
    child: dict | None,
    examiner: dict | None,
    *,
    tau_face: float,
) -> None:
    buf = buffers[tid]
    if child is not None:
        w = face_weight(child)
        hp = child.get("head_pose") or {}
        yaw = hp.get("yaw")
        pitch = hp.get("pitch")
        roll = hp.get("roll")
        mar = child.get("mar")
        buf["c_w"].append(w)
        buf["c_mar"].append(mar)
        buf["c_yaw"].append(yaw)
        buf["c_pitch"].append(pitch)
        buf["c_roll"].append(roll)
        buf["c_yaw_abs"].append(abs(yaw) if yaw is not None else None)

    if child is not None and examiner is not None:
        wc = face_weight(child)
        we = face_weight(examiner)
        w_dy = min(wc, we)
        if w_dy >= tau_face:
            hpc = child.get("head_pose") or {}
            hpe = examiner.get("head_pose") or {}
            yc, ye = hpc.get("yaw"), hpe.get("yaw")
            pc, pe = hpc.get("pitch"), hpe.get("pitch")
            yaw_diff = None
            if yc is not None and ye is not None:
                yaw_diff = abs(yc - ye)
            mutual = 0.0
            if (
                yc is not None
                and ye is not None
                and pc is not None
                and pe is not None
                and abs(yc - ye) < 30.0
                and abs(pc) < 25.0
                and abs(pe) < 25.0
            ):
                mutual = 1.0
            buf["d_w"].append(w_dy)
            buf["d_yaw_diff"].append(yaw_diff)
            buf["d_mutual"].append(mutual)
            buf["d_mar_c"].append(child.get("mar"))
            buf["d_mar_e"].append(examiner.get("mar"))
            buf["dyad_valid_frames"] += 1


def finalize_face(
    buf: dict[str, Any],
    task_dur: float,
    *,
    sample_interval: float = 0.25,
) -> dict[str, float]:
    w = buf["c_w"]
    # cross-corr needs paired finite mars
    mar_c = np.asarray(
        [a if a is not None else np.nan for a in buf["d_mar_c"]], dtype=np.float64
    )
    mar_e = np.asarray(
        [a if a is not None else np.nan for a in buf["d_mar_e"]], dtype=np.float64
    )
    mask = np.isfinite(mar_c) & np.isfinite(mar_e)
    if np.count_nonzero(mask) >= 8:
        cc = float(np.corrcoef(mar_c[mask], mar_e[mask])[0, 1])
    else:
        cc = float("nan")

    valid_time = buf["dyad_valid_frames"] * sample_interval
    return {
        "child_mar_wmean": weighted_mean(buf["c_mar"], w),
        "child_mar_wstd": weighted_std(buf["c_mar"], w),
        "child_mar_p90": weighted_quantile(buf["c_mar"], w, 0.90),
        "child_yaw_wmean": weighted_mean(buf["c_yaw"], w),
        "child_yaw_wstd": weighted_std(buf["c_yaw"], w),
        "child_pitch_wmean": weighted_mean(buf["c_pitch"], w),
        "child_pitch_wstd": weighted_std(buf["c_pitch"], w),
        "child_roll_wstd": weighted_std(buf["c_roll"], w),
        "child_yaw_abs_wmean": weighted_mean(buf["c_yaw_abs"], w),
        "dyad_yaw_diff_wmean": weighted_mean(buf["d_yaw_diff"], buf["d_w"]),
        "dyad_yaw_diff_wstd": weighted_std(buf["d_yaw_diff"], buf["d_w"]),
        "dyad_mutual_orient_frac": weighted_mean(buf["d_mutual"], buf["d_w"]),
        "dyad_mar_crosscorr": cc,
        "dyad_face_valid_time_frac": safe_div(valid_time, task_dur),
    }
