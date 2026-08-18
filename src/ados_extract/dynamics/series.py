"""Uniform-grid time series per role, with the scale and gap handling that
the dynamic features depend on.

The static features in `ados_seg.features` summarise instantaneous
quantities. Everything the literature reports as discriminative is instead
a property of how those quantities move: angular velocity and multiscale
entropy of head pose (Martin et al. 2018; Zhao et al. 2021), pixel distance
and instantaneous velocity of joints (Han et al. 2023), motion energy and
its cross-correlation between partners (Koehler et al. 2024). Those need a
time base, so this module provides one.

Three things have to be right before any derivative is taken.

Sampling. 99.9% of frames sit on a 0.25 s grid, but the remainder jump by
up to several seconds. A difference taken across such a jump is not a
velocity. Series are placed on an explicit uniform grid and derivatives are
only formed between adjacent grid cells that both carry data.

Scale. Keypoints are in pixels, and shoulder width ranges from about 32 to
187 px across the cohort, so an unnormalised pixel velocity would mostly
report how close to the camera someone sat. This is the same class of
artefact as the head-yaw one already found in this cohort, where absolute
yaw turned out to encode seating rather than behaviour. Distances are
divided by a rolling median of shoulder width, which tracks slow changes in
depth while leaving genuine movement in place.

Availability. The existing pipeline gates every feature on both faces being
visible, which costs 38% of frames. A head-movement feature for the child
needs the child, not the pair, so validity is tracked per role and the
paired mask is formed only where it is actually required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ados_extract.dynamics._support import face_weight
from ados_extract.dynamics._support import is_role_swap

GRID_SEC = 0.25
KEYPOINTS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
# A shoulder width below this many pixels is a failed detection, not a
# distant child; the 5th percentile across the cohort is about 32 px.
MIN_SHOULDER_PX = 15.0
SCALE_WINDOW_SEC = 30.0
# Head pose is reported in degrees. A jump this large inside 0.25 s is
# faster than a child can turn and marks a tracking failure.
MAX_ANGLE_JUMP_DEG = 120.0


@dataclass
class RoleSeries:
    """One participant's signals on the shared grid. NaN marks absence."""

    role: str
    t: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    mar: np.ndarray
    kp: dict[str, np.ndarray]  # name -> (n, 2) pixels
    scale: np.ndarray  # rolling shoulder width, pixels
    face_ok: np.ndarray
    pose_ok: np.ndarray
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t)

    def kp_norm(self, name: str) -> np.ndarray:
        """Keypoint in body-widths rather than pixels."""
        v = self.kp[name].copy()
        s = self.scale.reshape(-1, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return v / s


@dataclass
class SessionSeries:
    participant_id: str
    t: np.ndarray
    child: RoleSeries
    examiner: RoleSeries
    role_swap: np.ndarray
    both_face: np.ndarray

    @property
    def n(self) -> int:
        return len(self.t)


def _pick(instances: list[dict] | None, role: str) -> dict | None:
    for inst in instances or []:
        if inst.get("role") == role:
            return inst
    return None


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    """Centred rolling median that ignores NaN and never returns all-NaN."""
    n = len(x)
    out = np.full(n, np.nan)
    half = max(1, win // 2)
    finite = np.isfinite(x)
    if not finite.any():
        return out
    fallback = float(np.nanmedian(x))
    idx = np.flatnonzero(finite)
    for i in range(n):
        lo, hi = i - half, i + half + 1
        j = idx[(idx >= lo) & (idx < hi)]
        out[i] = float(np.median(x[j])) if len(j) >= 3 else np.nan
    out[~np.isfinite(out)] = fallback
    return out


def _clean_angle(a: np.ndarray) -> tuple[np.ndarray, float]:
    """NaN out impossible frame-to-frame jumps; report how many."""
    a = a.copy()
    d = np.abs(np.diff(a))
    bad = np.flatnonzero(d > MAX_ANGLE_JUMP_DEG)
    # Blank the later member of each offending pair.
    a[bad + 1] = np.nan
    n_fin = int(np.isfinite(a).sum())
    return a, float(len(bad) / n_fin) if n_fin else np.nan


def build_session_series(
    timeline: Sequence[dict[str, Any]],
    participant_id: str,
    *,
    tau_face: float = 0.25,
    tau_pose: float = 0.25,
    reject_role_swaps: bool = True,
) -> SessionSeries:
    """Place the session on a uniform grid and split it by role."""
    times = np.array([float(f["time_sec"]) for f in timeline], dtype=float)
    if len(times) == 0:
        raise ValueError("empty timeline")
    t0, t1 = float(times[0]), float(times[-1])
    n = int(round((t1 - t0) / GRID_SEC)) + 1
    grid = t0 + GRID_SEC * np.arange(n)
    slot = np.rint((times - t0) / GRID_SEC).astype(int)
    ok = (slot >= 0) & (slot < n)

    store = {
        role: {
            "yaw": np.full(n, np.nan),
            "pitch": np.full(n, np.nan),
            "roll": np.full(n, np.nan),
            "mar": np.full(n, np.nan),
            "face_ok": np.zeros(n, bool),
            "pose_ok": np.zeros(n, bool),
            "kp": {k: np.full((n, 2), np.nan) for k in KEYPOINTS},
        }
        for role in ("child", "examiner")
    }
    both_face = np.zeros(n, bool)
    swap = np.zeros(n, bool)
    prev_ok: tuple[dict, dict] | None = None

    for fr, s, good in zip(timeline, slot, ok):
        if not good:
            continue
        faces, poses = fr.get("face") or [], fr.get("pose") or []
        cf, ef = _pick(faces, "child"), _pick(faces, "examiner")
        pair = (
            cf is not None
            and ef is not None
            and face_weight(cf) >= tau_face
            and face_weight(ef) >= tau_face
        )
        both_face[s] = pair
        if pair and reject_role_swaps and prev_ok is not None:
            swap[s] = is_role_swap(prev_ok[0], prev_ok[1], cf, ef)
        prev_ok = (cf, ef) if pair else None

        for role in ("child", "examiner"):
            st = store[role]
            f = _pick(faces, role)
            if f is not None and face_weight(f) >= tau_face:
                hp = f.get("head_pose") or {}
                st["yaw"][s] = hp.get("yaw", np.nan)
                st["pitch"][s] = hp.get("pitch", np.nan)
                st["roll"][s] = hp.get("roll", np.nan)
                mar = f.get("mar")
                st["mar"][s] = mar if mar is not None else np.nan
                st["face_ok"][s] = True
            p = _pick(poses, role)
            if p is None:
                continue
            if float(p.get("pose_quality") or 0.0) < tau_pose:
                continue
            kpx = p.get("keypoints_px") or {}
            ls, rs = kpx.get("left_shoulder"), kpx.get("right_shoulder")
            if not (ls and rs):
                continue
            width = float(np.hypot(ls["x"] - rs["x"], ls["y"] - rs["y"]))
            if not np.isfinite(width) or width < MIN_SHOULDER_PX:
                continue
            st["pose_ok"][s] = True
            for k in KEYPOINTS:
                v = kpx.get(k)
                if v and float(v.get("visibility", 1.0)) > 0.0:
                    st["kp"][k][s] = (float(v["x"]), float(v["y"]))

    win = int(round(SCALE_WINDOW_SEC / GRID_SEC))
    roles: dict[str, RoleSeries] = {}
    for role in ("child", "examiner"):
        st = store[role]
        ls, rs = st["kp"]["left_shoulder"], st["kp"]["right_shoulder"]
        raw_scale = np.hypot(ls[:, 0] - rs[:, 0], ls[:, 1] - rs[:, 1])
        raw_scale[raw_scale < MIN_SHOULDER_PX] = np.nan
        scale = _rolling_median(raw_scale, win)
        yaw, r_yaw = _clean_angle(st["yaw"])
        pitch, r_pitch = _clean_angle(st["pitch"])
        roll, r_roll = _clean_angle(st["roll"])
        roles[role] = RoleSeries(
            role=role,
            t=grid,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            mar=st["mar"],
            kp=st["kp"],
            scale=scale,
            face_ok=st["face_ok"],
            pose_ok=st["pose_ok"],
            diagnostics={
                "face_ok_frac": float(st["face_ok"].mean()),
                "pose_ok_frac": float(st["pose_ok"].mean()),
                "scale_px_median": float(np.nanmedian(raw_scale)),
                "angle_jump_frac": float(np.nanmean([r_yaw, r_pitch, r_roll])),
            },
        )

    return SessionSeries(
        participant_id=participant_id,
        t=grid,
        child=roles["child"],
        examiner=roles["examiner"],
        role_swap=swap,
        both_face=both_face,
    )


def adjacent_diff(x: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """First difference, defined only between neighbouring populated cells.

    Returns an array of length n-1. A cell is NaN when either endpoint is
    absent, which keeps a gap in the recording from being read as a large
    displacement.
    """
    x = np.asarray(x, float)
    d = np.diff(x)
    if valid is not None:
        v = np.asarray(valid, bool)
        d[~(v[:-1] & v[1:])] = np.nan
    return d


def masked_series(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(x, float).copy()
    out[~np.asarray(mask, bool)] = np.nan
    return out


def contiguous_runs(mask: np.ndarray, min_len: int = 2) -> list[tuple[int, int]]:
    """Half-open index ranges of consecutive True cells."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(m.astype(int))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        ends = ends + [len(m)]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]
