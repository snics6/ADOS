"""Leave-one-out group centering (preterm / other). Used once after Exp1–4."""

from __future__ import annotations

import numpy as np
import pandas as pd


def loo_center(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract same-group mean, excluding the person. Group size 1 → NaN."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    out = np.full(v.shape, np.nan, dtype=float)
    for name in pd.unique(g):
        idx = np.flatnonzero(g == name)
        if idx.size < 2:
            continue
        s = float(np.nansum(v[idx]))
        n_ok = int(np.isfinite(v[idx]).sum())
        if n_ok < 2:
            continue
        # LOO mean uses finite others only
        for i in idx:
            if not np.isfinite(v[i]):
                continue
            out[i] = v[i] - (s - v[i]) / (n_ok - 1)
    return out


def residualize_frame(fr: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = fr.copy()
    g = out["group"].to_numpy()
    for c in list(dict.fromkeys(cols)):
        ser = out[c]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        out[c] = loo_center(pd.to_numeric(ser, errors="coerce").to_numpy(dtype=float), g)
    return out


def group_counts(groups: np.ndarray | list) -> dict[str, int]:
    s = pd.Series(groups).astype(str)
    return {
        "n_preterm": int((s == "preterm").sum()),
        "n_other": int((s == "other").sum()),
    }


def small_group(n_preterm: int, n_other: int, thresh: int = 10) -> bool:
    return n_preterm < thresh or n_other < thresh


def classify(
    old_stat: float,
    new_stat: float,
    old_sig: bool,
    new_sig: bool,
) -> dict[str, bool | str]:
    """Collapsed = sign flip or lost FDR. Weakened = same sign, |stat| ≤ half."""
    o = float(old_stat) if np.isfinite(old_stat) else float("nan")
    n = float(new_stat) if np.isfinite(new_stat) else float("nan")
    if not np.isfinite(n):
        return {
            "sign_flip": False,
            "lost_fdr": bool(old_sig),
            "collapsed": bool(old_sig),
            "weakened": False,
            "verdict": "欠け",
        }
    so = 0 if o == 0 else (1 if o > 0 else -1)
    sn = 0 if n == 0 else (1 if n > 0 else -1)
    sign_flip = so != 0 and sn != 0 and so != sn
    lost_fdr = bool(old_sig) and not bool(new_sig)
    collapsed = bool(sign_flip or lost_fdr)
    weakened = (
        (not collapsed)
        and so == sn
        and so != 0
        and abs(n) <= 0.5 * abs(o)
    )
    if collapsed:
        verdict = "崩れた"
    elif weakened:
        verdict = "弱まった"
    else:
        verdict = "崩れず"
    return {
        "sign_flip": sign_flip,
        "lost_fdr": lost_fdr,
        "collapsed": collapsed,
        "weakened": weakened,
        "verdict": verdict,
    }
