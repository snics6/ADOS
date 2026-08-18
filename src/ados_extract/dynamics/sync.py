"""Interpersonal coordination between child and examiner.

Koehler et al. (2024, Sci Rep) classified autism from ADOS-2 recordings
using motion synchrony alone, at a balanced accuracy of 63.4% on 94 dyads,
which makes this the closest published comparison for the present data.
The standard pipeline (Ramseyer & Tschacher; Altmann; Dunbar et al. 2022)
is to reduce each person to a motion-energy series, then quantify coupling
with windowed cross-lagged correlation and with cross-recurrence
quantification.

Both are reported against a surrogate baseline. Two unrelated series will
correlate simply because both people move in bursts of similar length, so
the quantity of interest is the excess over what shuffled pairings of the
same windows produce. Reporting raw synchrony without that subtraction is
the standard way to find synchrony that is not there.
"""

from __future__ import annotations

import numpy as np

from ados_extract.dynamics.series import GRID_SEC, SessionSeries, contiguous_runs

# Tasks run 2 to 4 minutes and the longest stretch with both people
# tracked is typically 60 to 85 s, so a 30 s window would yield three
# estimates per cell. Fifteen seconds still spans several movement bursts.
WIN_SEC = 15.0
STEP_SEC = 5.0
MAX_LAG_SEC = 3.0
N_SURROGATE = 20
# Radius as a quantile of the pairwise distances, so the recurrence rate is
# comparable between dyads regardless of how much each one moved.
CRQA_RADIUS_Q = 0.05
CRQA_EMBED = 3
CRQA_DELAY = 4  # 1 s at 4 Hz
CRQA_MAX_N = 1500
CRQA_MAX_RUNS = 8


def motion_energy(ss: SessionSeries, role: str, mask: np.ndarray) -> np.ndarray:
    """Mean normalised joint speed, the series MEA would produce."""
    from ados_extract.dynamics.kinematics import MOVING_KP, _speed_norm

    rs = getattr(ss, role)
    valid = mask & rs.pose_ok
    sp = np.vstack([_speed_norm(rs, k, valid) for k in MOVING_KP])
    out = np.nanmean(sp, axis=0)
    out[np.all(~np.isfinite(sp), axis=0)] = np.nan
    # A difference is one shorter than its source; pad the front so the
    # series stays aligned with the frame grid and its masks.
    return np.concatenate(([np.nan], out))


def _peak_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, int]:
    """Largest absolute correlation over lags, and the lag achieving it."""
    best_r, best_l = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = a[-lag:], b[: len(b) + lag]
        elif lag > 0:
            x, y = a[: len(a) - lag], b[lag:]
        else:
            x, y = a, b
        if len(x) < 16:
            continue
        sx, sy = np.std(x), np.std(y)
        if sx <= 0 or sy <= 0:
            continue
        r = float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))
        if abs(r) > abs(best_r):
            best_r, best_l = r, lag
    return best_r, best_l


def windowed_sync(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray, prefix: str
) -> dict[str, float]:
    """Windowed cross-lagged correlation with a shuffled-window baseline."""
    out: dict[str, float] = {}
    w = int(WIN_SEC / GRID_SEC)
    step = int(STEP_SEC / GRID_SEC)
    max_lag = int(MAX_LAG_SEC / GRID_SEC)
    ok = mask & np.isfinite(a) & np.isfinite(b)

    wins: list[tuple[np.ndarray, np.ndarray]] = []
    for s, e in contiguous_runs(ok, w):
        for i in range(s, e - w + 1, step):
            wins.append((a[i : i + w], b[i : i + w]))
    if len(wins) < 3:
        return out

    peaks, lags = [], []
    for x, y in wins:
        r, l = _peak_xcorr(x, y, max_lag)
        peaks.append(abs(r))
        lags.append(l)
    peaks_a, lags_a = np.array(peaks), np.array(lags)

    rng = np.random.default_rng(0)
    sur = []
    for _ in range(N_SURROGATE):
        j = rng.permutation(len(wins))
        for k, jj in enumerate(j):
            if jj == k:
                continue
            r, _l = _peak_xcorr(wins[k][0], wins[jj][1], max_lag)
            sur.append(abs(r))
    sur_mean = float(np.mean(sur)) if sur else np.nan

    out[f"{prefix}_sync_peak"] = float(peaks_a.mean())
    out[f"{prefix}_sync_peak_sd"] = float(peaks_a.std())
    out[f"{prefix}_sync_surrogate"] = sur_mean
    out[f"{prefix}_sync_excess"] = float(peaks_a.mean() - sur_mean)
    out[f"{prefix}_sync_strong_frac"] = float((peaks_a > sur_mean + 0.1).mean())
    out[f"{prefix}_sync_lag_abs"] = float(np.mean(np.abs(lags_a)) * GRID_SEC)
    # Positive means the child's movement precedes the examiner's.
    out[f"{prefix}_sync_lead"] = float(np.mean(np.sign(lags_a)))
    out[f"{prefix}_sync_n_windows"] = float(len(wins))
    return out


def _embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    n = len(x) - (m - 1) * tau
    if n <= 0:
        return np.empty((0, m))
    return np.stack([x[i * tau : i * tau + n] for i in range(m)], axis=1)


def _crqa_one(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    # Standardise so the radius means the same thing for every dyad.
    x = (x - x.mean()) / (x.std() or 1.0)
    y = (y - y.mean()) / (y.std() or 1.0)
    ex, ey = _embed(x, CRQA_EMBED, CRQA_DELAY), _embed(y, CRQA_EMBED, CRQA_DELAY)
    if len(ex) < 100 or len(ey) < 100:
        return None
    d = np.sqrt(((ex[:, None, :] - ey[None, :, :]) ** 2).sum(-1))
    R = d <= float(np.quantile(d, CRQA_RADIUS_Q))
    npts = float(R.sum())
    if npts <= 0:
        return None
    out = {"rr": float(R.mean())}
    dl = np.array(_line_lengths(R, diagonal=True) or [0])
    vl = np.array(_line_lengths(R, diagonal=False) or [0])
    if (dl >= 2).any():
        out["det"] = float(dl[dl >= 2].sum() / npts)
        out["maxline"] = float(dl.max())
        out["meanline"] = float(dl[dl >= 2].mean())
        h, _ = np.histogram(dl[dl >= 2], bins=np.arange(2, max(3, dl.max() + 2)))
        pr = h[h > 0] / h.sum() if h.sum() else np.array([1.0])
        out["entropy"] = float(-np.sum(pr * np.log(pr)))
    if (vl >= 2).any():
        out["lam"] = float(vl[vl >= 2].sum() / npts)
        out["tt"] = float(vl[vl >= 2].mean())
    return out


def crqa(a: np.ndarray, b: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    """Cross-recurrence quantification, pooled over usable stretches.

    Tracking dropouts cap a single stretch at roughly 60 to 85 s however
    long the window is, so a session cannot supply a longer run than a task
    can, only more of them. Each run is quantified separately and combined
    by length, which is what turns extra recording time into a steadier
    estimate.
    """
    ok = mask & np.isfinite(a) & np.isfinite(b)
    runs = sorted(contiguous_runs(ok, 120), key=lambda r: r[1] - r[0], reverse=True)
    if not runs:
        return {}
    vals: list[tuple[float, dict[str, float]]] = []
    for s, e in runs[:CRQA_MAX_RUNS]:
        x, y = a[s:e], b[s:e]
        if len(x) > CRQA_MAX_N:
            x, y = x[:CRQA_MAX_N], y[:CRQA_MAX_N]
        r = _crqa_one(x, y)
        if r is not None:
            vals.append((float(len(x)), r))
    if not vals:
        return {}
    keys = {k for _w, r in vals for k in r}
    out: dict[str, float] = {}
    for k in keys:
        num = sum(w * r[k] for w, r in vals if k in r)
        den = sum(w for w, r in vals if k in r)
        out[f"{prefix}_crqa_{k}"] = float(num / den) if den else np.nan
    out[f"{prefix}_crqa_n_runs"] = float(len(vals))
    return out


def _line_lengths(R: np.ndarray, diagonal: bool) -> list[int]:
    lens: list[int] = []
    if diagonal:
        n, m = R.shape
        for off in range(-(n - 1), m):
            lens.extend(_runs(np.diagonal(R, offset=off)))
    else:
        for col in range(R.shape[1]):
            lens.extend(_runs(R[:, col]))
    return lens


def _runs(v: np.ndarray) -> list[int]:
    if v.size == 0 or not v.any():
        return []
    d = np.diff(np.concatenate(([0], v.view(np.int8), [0])))
    return list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))


def sync_features(ss: SessionSeries, mask: np.ndarray) -> dict[str, float]:
    """Movement and head-orientation coupling between the two people."""
    out: dict[str, float] = {}
    mc = motion_energy(ss, "child", mask)
    me = motion_energy(ss, "examiner", mask)
    out.update(windowed_sync(mc, me, mask, "dyad_motion"))
    out.update(crqa(mc, me, mask, "dyad_motion"))

    both = mask & ss.child.face_ok & ss.examiner.face_ok & ~ss.role_swap
    for ang in ("yaw", "pitch"):
        a = np.asarray(getattr(ss.child, ang), float)
        b = np.asarray(getattr(ss.examiner, ang), float)
        out.update(windowed_sync(a, b, both, f"dyad_{ang}"))
    out.update(crqa(np.asarray(ss.child.yaw, float),
                    np.asarray(ss.examiner.yaw, float), both, "dyad_yaw"))

    # How much of the pair's movement is the child's, and whether the two
    # rise and fall together at all.
    ok = mask & np.isfinite(mc) & np.isfinite(me)
    if ok.sum() > 64:
        tot = float(np.nansum(mc[ok]) + np.nansum(me[ok]))
        if tot > 0:
            out["dyad_motion_share_child"] = float(np.nansum(mc[ok]) / tot)
        out["dyad_motion_corr"] = float(np.corrcoef(mc[ok], me[ok])[0, 1])
    return out
