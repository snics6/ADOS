"""Head-pose and body kinematic descriptors, per role.

Head pose. Martin et al. (2018, Mol Autism) separated autistic from
non-autistic children by the root mean square of yaw, pitch and roll
(angular displacement) and by angular velocity, with the differences
carried by yaw and roll rather than pitch. Zhao et al. (2021, Autism Res)
added rotation range, amount of rotation per minute, and multiscale entropy
as an index of movement stereotypy. Perochon et al. (Duke) report rate,
acceleration and multiscale entropy in toddlers.

Body. Han et al. (2023, Sci Rep), working from ADOS recordings, found that
accumulated pixel distance and instantaneous pixel velocity of the wrists
tracked clinician activity-level ratings at rho above 0.6. Expressive-
gesture work (Camurri and colleagues) contributes quantity of motion,
contraction index and the distinction between movement bursts and pauses.

Two properties matter for this cohort specifically. Velocity and RMS are
invariant to an additive offset, so they are untouched by the seating
geometry that made absolute yaw a measure of where the child sat. And
dividing by shoulder width removes the camera distance that would otherwise
make anyone sitting closer appear more active.
"""

from __future__ import annotations

import numpy as np

from ados_extract.dynamics.series import (
    GRID_SEC,
    RoleSeries,
    adjacent_diff,
    contiguous_runs,
)

ANGLES = ("yaw", "pitch", "roll")
MOVING_KP = ("nose", "left_wrist", "right_wrist", "left_elbow", "right_elbow")
# A wrist crossing a fifth of a shoulder width per second is moving; below
# that the person is holding still.
STILL_THRESH = 0.20
BURST_THRESH = 1.00
MIN_RUN = 8  # 2 s of consecutive data before a derivative is worth taking


def _stats(v: np.ndarray, prefix: str) -> dict[str, float]:
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_rms": float(np.sqrt(np.mean(v**2))),
        f"{prefix}_med": float(np.median(v)),
        f"{prefix}_p90": float(np.percentile(v, 90)),
        f"{prefix}_iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
    }


def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """SampEn on a contiguous, finite series. Higher means less predictable."""
    x = np.asarray(x, float)
    n = len(x)
    if n < m + 20:
        return np.nan
    sd = float(np.std(x))
    if sd <= 0:
        return np.nan
    r = r_frac * sd

    def _count(mm: int) -> int:
        if n - mm < 2:
            return 0
        idx = np.arange(n - mm + 1)
        emb = x[idx[:, None] + np.arange(mm)]
        total = 0
        for i in range(len(emb) - 1):
            d = np.max(np.abs(emb[i + 1 :] - emb[i]), axis=1)
            total += int((d <= r).sum())
        return total

    a, b = _count(m + 1), _count(m)
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


def multiscale_entropy(
    x: np.ndarray, mask: np.ndarray, scales: tuple[int, ...] = (1, 2, 3)
) -> dict[int, float]:
    """SampEn on coarse-grained versions, pooled over contiguous runs.

    Coarse-graining averages non-overlapping blocks, so scale 2 asks whether
    the signal is predictable at 0.5 s rather than 0.25 s.
    """
    out: dict[int, float] = {}
    runs = [
        x[s:e]
        for s, e in contiguous_runs(mask & np.isfinite(x), MIN_RUN)
        if e - s >= 48
    ]
    if not runs:
        return {s: np.nan for s in scales}
    # Cap the work: entropy is quadratic in length and the longest runs
    # dominate the estimate anyway.
    runs = sorted(runs, key=len, reverse=True)[:6]
    for sc in scales:
        vals = []
        for seg in runs:
            k = len(seg) // sc
            if k < 48:
                continue
            cg = seg[: k * sc].reshape(k, sc).mean(axis=1) if sc > 1 else seg
            v = sample_entropy(cg[:2000])
            if np.isfinite(v):
                vals.append(v)
        out[sc] = float(np.mean(vals)) if vals else np.nan
    return out


def head_features(rs: RoleSeries, mask: np.ndarray) -> dict[str, float]:
    """Displacement, speed and predictability of head rotation."""
    out: dict[str, float] = {}
    p = rs.role
    valid = mask & rs.face_ok
    dur_min = float(valid.sum()) * GRID_SEC / 60.0
    if dur_min <= 0.05:
        return out

    speeds = []
    for ang in ANGLES:
        a = np.asarray(getattr(rs, ang), float).copy()
        a[~valid] = np.nan
        fin = a[np.isfinite(a)]
        if len(fin) < 8:
            continue
        # Displacement about the person's own centre: an offset here is
        # seating, not behaviour.
        out[f"{p}_{ang}_rms_disp"] = float(np.sqrt(np.mean((fin - fin.mean()) ** 2)))
        out[f"{p}_{ang}_range"] = float(
            np.percentile(fin, 95) - np.percentile(fin, 5)
        )
        d = adjacent_diff(a, np.isfinite(a))
        v = np.abs(d) / GRID_SEC
        out.update(_stats(v, f"{p}_{ang}_angvel"))
        out[f"{p}_{ang}_rot_per_min"] = float(np.nansum(np.abs(d)) / dur_min)
        acc = np.abs(adjacent_diff(d)) / GRID_SEC**2
        out.update(_stats(acc, f"{p}_{ang}_angacc"))
        # How often the rotation reverses: a proxy for oscillatory motion
        # such as nodding or shaking.
        s = d[np.isfinite(d)]
        if len(s) > 8:
            out[f"{p}_{ang}_reversal_rate"] = float(
                np.mean(np.diff(np.sign(s)) != 0) / GRID_SEC
            )
        speeds.append(np.abs(d))

    if len(speeds) == 3:
        comb = np.sqrt(np.nansum(np.stack(speeds) ** 2, axis=0)) / GRID_SEC
        comb[~np.isfinite(np.stack(speeds)).all(axis=0)] = np.nan
        out.update(_stats(comb, f"{p}_head_speed"))

    for ang in ("yaw", "pitch"):
        a = np.asarray(getattr(rs, ang), float)
        mse = multiscale_entropy(a, valid)
        for sc, v in mse.items():
            out[f"{p}_{ang}_sampen_s{sc}"] = v
        f = [v for v in mse.values() if np.isfinite(v)]
        if len(f) >= 2:
            out[f"{p}_{ang}_mse_slope"] = float(f[-1] - f[0])
    return out


def _speed_norm(rs: RoleSeries, name: str, valid: np.ndarray) -> np.ndarray:
    """Keypoint speed in shoulder-widths per second."""
    v = rs.kp_norm(name)
    ok = valid & np.isfinite(v).all(axis=1)
    dx = adjacent_diff(v[:, 0], ok)
    dy = adjacent_diff(v[:, 1], ok)
    return np.hypot(dx, dy) / GRID_SEC


def body_features(rs: RoleSeries, mask: np.ndarray) -> dict[str, float]:
    """Joint speed, motion quantity, bursts, asymmetry and posture."""
    out: dict[str, float] = {}
    p = rs.role
    valid = mask & rs.pose_ok
    n_valid = int(valid.sum())
    dur_min = n_valid * GRID_SEC / 60.0
    if dur_min <= 0.05:
        return out
    out[f"{p}_pose_frames"] = float(n_valid)

    path_total = np.zeros(0)
    per_kp: dict[str, np.ndarray] = {}
    for k in MOVING_KP:
        sp = _speed_norm(rs, k, valid)
        per_kp[k] = sp
        out.update(_stats(sp, f"{p}_{k}_speed"))
        out[f"{p}_{k}_path_per_min"] = float(np.nansum(sp) * GRID_SEC / dur_min)
        acc = np.abs(adjacent_diff(sp)) / GRID_SEC
        out.update(_stats(acc, f"{p}_{k}_accel"))

    stack = np.vstack([per_kp[k] for k in MOVING_KP])
    qom = np.nanmean(stack, axis=0)
    qom[np.all(~np.isfinite(stack), axis=0)] = np.nan
    out.update(_stats(qom, f"{p}_qom"))
    out[f"{p}_qom_per_min"] = float(np.nansum(qom) * GRID_SEC / dur_min)

    fin = qom[np.isfinite(qom)]
    if len(fin) > 16:
        out[f"{p}_still_frac"] = float((fin < STILL_THRESH).mean())
        burst = fin >= BURST_THRESH
        out[f"{p}_burst_frac"] = float(burst.mean())
        runs = contiguous_runs(burst, 1)
        out[f"{p}_burst_rate_per_min"] = float(len(runs) / dur_min)
        out[f"{p}_burst_dur_mean"] = (
            float(np.mean([e - s for s, e in runs]) * GRID_SEC) if runs else 0.0
        )
        # Spectral shape of the motion signal. The 0.25 s grid puts the
        # Nyquist limit at 2 Hz, so the fast end of the stereotypy band
        # reported in the literature is not observable here.
        seg = max(contiguous_runs(np.isfinite(qom), 64), key=lambda r: r[1] - r[0],
                  default=None)
        if seg is not None:
            y = fin_seg = qom[seg[0] : seg[1]]
            y = y - y.mean()
            f, P = _welch(y, 1.0 / GRID_SEC)
            tot = float(P.sum())
            if tot > 0:
                out[f"{p}_pow_lo"] = float(P[(f >= 0.1) & (f < 0.5)].sum() / tot)
                out[f"{p}_pow_mid"] = float(P[(f >= 0.5) & (f < 1.0)].sum() / tot)
                out[f"{p}_pow_hi"] = float(P[(f >= 1.0) & (f <= 2.0)].sum() / tot)
                # Skip the lowest bins: almost all the power sits in the
                # slow drift of where the body happens to be.
                band = f >= 0.2
                if band.any():
                    out[f"{p}_dom_freq"] = float(f[band][int(np.argmax(P[band]))])
                q = P / tot
                q = q[q > 0]
                out[f"{p}_spec_entropy"] = float(-np.sum(q * np.log(q)) / np.log(len(q)))
            out[f"{p}_qom_sampen"] = sample_entropy(fin_seg[:2000])

    lw = np.nansum(per_kp["left_wrist"])
    rw = np.nansum(per_kp["right_wrist"])
    if lw + rw > 0:
        out[f"{p}_wrist_asym"] = float(abs(lw - rw) / (lw + rw))

    # Posture: how far the upper body spreads, and how much the head drifts.
    xs, ys = [], []
    for k in ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "nose"):
        v = rs.kp_norm(k)
        xs.append(v[:, 0])
        ys.append(v[:, 1])
    X, Y = np.vstack(xs), np.vstack(ys)
    okc = valid & np.isfinite(X).all(axis=0) & np.isfinite(Y).all(axis=0)
    if okc.sum() > 16:
        w = X[:, okc].max(axis=0) - X[:, okc].min(axis=0)
        h = Y[:, okc].max(axis=0) - Y[:, okc].min(axis=0)
        out[f"{p}_contraction_mean"] = float(np.mean(w * h))
        out[f"{p}_contraction_sd"] = float(np.std(w * h))
    nose = rs.kp_norm("nose")
    okn = valid & np.isfinite(nose).all(axis=1)
    if okn.sum() > 16:
        out[f"{p}_sway_x"] = float(np.std(nose[okn, 0]))
        out[f"{p}_sway_y"] = float(np.std(nose[okn, 1]))
    return out


def _welch(y: np.ndarray, fs: float, nperseg: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Welch estimate; avoids a scipy import in the hot loop."""
    n = len(y)
    nperseg = min(nperseg, n)
    step = max(1, nperseg // 2)
    win = np.hanning(nperseg)
    segs = [y[i : i + nperseg] for i in range(0, n - nperseg + 1, step)]
    if not segs:
        segs = [np.pad(y, (0, nperseg - n))]
    P = np.zeros(nperseg // 2 + 1)
    for s in segs:
        P += np.abs(np.fft.rfft((s - s.mean()) * win)) ** 2
    P /= len(segs)
    return np.fft.rfftfreq(nperseg, 1.0 / fs), P
