"""Coupling between what a person does with their body and who is talking.

Listener feedback, chiefly nodding while the partner holds the floor, is a
staple of social signal processing and is not captured by either the motion
or the speech features on their own. The same applies to gesture, which in
typical interaction rides on the speaker's own utterances: how much of a
person's movement falls inside their own speech rather than the partner's
separates a gesturing speaker from someone who moves independently of the
conversation.

Each quantity is expressed as a contrast between two states of the floor,
so the overall amount a person moves or talks divides out.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ados_extract.dynamics.series import GRID_SEC, SessionSeries, adjacent_diff

NOD_MIN_DEG = 3.0


def speech_masks(
    speech_segments: Sequence[dict[str, Any]], t: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-frame indicator of who holds the floor."""
    out = {r: np.zeros(len(t), bool) for r in ("child", "examiner")}
    for u in speech_segments:
        r = u.get("speaker")
        if r not in out:
            continue
        out[r] |= (t >= float(u["start"]) - GRID_SEC / 2) & (
            t < float(u["end"]) + GRID_SEC / 2
        )
    out["silence"] = ~(out["child"] | out["examiner"])
    return out


def _nod_rate(pitch: np.ndarray, sel: np.ndarray) -> float:
    """Reversals of vertical head motion per second, above a size floor."""
    d = adjacent_diff(pitch, np.isfinite(pitch))
    s = np.concatenate(([np.nan], d))
    s[~sel] = np.nan
    big = s[np.isfinite(s) & (np.abs(s) >= NOD_MIN_DEG)]
    if len(big) < 6:
        return np.nan
    return float(np.mean(np.diff(np.sign(big)) != 0) / GRID_SEC)


def crossmodal_features(
    ss: SessionSeries,
    speech_segments: Sequence[dict[str, Any]],
    mask: np.ndarray,
) -> dict[str, float]:
    from ados_extract.dynamics.sync import motion_energy

    out: dict[str, float] = {}
    sm = speech_masks(speech_segments, ss.t)
    if not mask.any():
        return out

    for role, other in (("child", "examiner"), ("examiner", "child")):
        rs = getattr(ss, role)
        me = motion_energy(ss, role, mask)
        own = mask & sm[role] & np.isfinite(me)
        par = mask & sm[other] & np.isfinite(me)
        sil = mask & sm["silence"] & np.isfinite(me)
        if own.sum() > 16 and par.sum() > 16:
            a, b = float(np.nanmean(me[own])), float(np.nanmean(me[par]))
            out[f"{role}_motion_own_speech"] = a
            out[f"{role}_motion_partner_speech"] = b
            out[f"{role}_motion_speech_ratio"] = float(a / b) if b > 0 else np.nan
        if sil.sum() > 16 and par.sum() > 16:
            c = float(np.nanmean(me[sil]))
            out[f"{role}_motion_silence"] = c

        pv = mask & rs.face_ok
        nod_par = _nod_rate(np.asarray(rs.pitch, float), pv & sm[other])
        nod_own = _nod_rate(np.asarray(rs.pitch, float), pv & sm[role])
        out[f"{role}_nod_rate_listening"] = nod_par
        out[f"{role}_nod_rate_speaking"] = nod_own
        if np.isfinite(nod_par) and np.isfinite(nod_own):
            out[f"{role}_nod_listen_excess"] = float(nod_par - nod_own)

        m = np.asarray(rs.mar, float)
        sel_o, sel_p = pv & sm[role], pv & sm[other]
        if np.isfinite(m[sel_o]).sum() > 16 and np.isfinite(m[sel_p]).sum() > 16:
            # Mouth opening should rise while speaking; if it does not, the
            # speaker labels and the face track disagree.
            out[f"{role}_mar_speak_minus_listen"] = float(
                np.nanmean(m[sel_o]) - np.nanmean(m[sel_p])
            )
    return out
