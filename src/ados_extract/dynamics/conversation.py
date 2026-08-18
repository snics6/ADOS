"""Conversational timing, from utterance and word boundaries alone.

Bone et al. (Interspeech 2013, 2016) and Ochi et al. (2019, PLoS ONE) both
report that children with greater autism severity speak less, in shorter
turns, more slowly, with longer pauses and longer response latencies, and
that the psychologist mirrors this with more pausing and more latency of
her own. Ochi et al. reached 89% diagnostic accuracy from turn-taking gaps
and pause proportion.

The acoustic half of that literature (F0 contour, intensity, jitter,
shimmer) needs the waveform, and no audio accompanies this cohort. The
timing half does not: floor transfer offset, turn length, intra-turn pause,
overlap and articulation rate all follow from boundaries, which is what the
word-level transcript provides.

Word identity is used only for counting. Japanese ASR error rate on this
material is unmeasured, so anything that would depend on the words being
right is left out.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Consecutive utterances by one speaker closer than this belong to one turn.
TURN_MERGE_SEC = 1.0
# A turn starting after this much silence is treated as newly initiated
# rather than as a reply.
INITIATION_SILENCE_SEC = 2.0
BACKCHANNEL_SEC = 1.0
ROLES = ("child", "examiner")


def _clip(segs: Sequence[dict], t0: float, t1: float) -> list[dict]:
    out = []
    for s in segs:
        a, b = max(float(s["start"]), t0), min(float(s["end"]), t1)
        if b - a <= 0.05:
            continue
        d = dict(s)
        d["start"], d["end"] = a, b
        if s.get("words"):
            d["words"] = [w for w in s["words"]
                          if a - 0.05 <= float(w.get("start", a)) <= b + 0.05]
        out.append(d)
    return sorted(out, key=lambda x: x["start"])


def _turns(utts: list[dict]) -> list[dict]:
    """Group neighbouring utterances by the same speaker into turns."""
    turns: list[dict] = []
    for u in utts:
        if (
            turns
            and turns[-1]["speaker"] == u["speaker"]
            and u["start"] - turns[-1]["end"] <= TURN_MERGE_SEC
        ):
            t = turns[-1]
            t["pauses"].append(u["start"] - t["end"])
            t["end"] = u["end"]
            t["n_words"] += len(u.get("words") or [])
            t["speech"] += u["end"] - u["start"]
            continue
        turns.append(
            {
                "speaker": u["speaker"],
                "start": u["start"],
                "end": u["end"],
                "n_words": len(u.get("words") or []),
                "speech": u["end"] - u["start"],
                "pauses": [],
            }
        )
    return turns


def _union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    tot, cs, ce = 0.0, iv[0][0], iv[0][1]
    for a, b in iv[1:]:
        if a > ce:
            tot += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return tot + (ce - cs)


def _desc(v: list[float], prefix: str) -> dict[str, float]:
    a = np.array([x for x in v if np.isfinite(x)], float)
    if len(a) < 3:
        return {}
    return {
        f"{prefix}_med": float(np.median(a)),
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_iqr": float(np.subtract(*np.percentile(a, [75, 25]))),
        f"{prefix}_p90": float(np.percentile(a, 90)),
    }


def conversation_features(
    speech_segments: Sequence[dict[str, Any]], t0: float, t1: float
) -> dict[str, float]:
    """Turn-taking timing for both speakers over [t0, t1)."""
    out: dict[str, float] = {}
    dur = float(t1 - t0)
    if dur <= 1.0:
        return out
    utts = [u for u in _clip(speech_segments, t0, t1) if u.get("speaker") in ROLES]
    if len(utts) < 4:
        return out
    turns = _turns(utts)
    out["dyad_n_turns"] = float(len(turns))
    out["dyad_turn_rate_per_min"] = float(len(turns) / (dur / 60.0))

    spoken = {r: [(u["start"], u["end"]) for u in utts if u["speaker"] == r]
              for r in ROLES}
    tot_speech = {r: _union(spoken[r]) for r in ROLES}
    any_speech = _union(spoken["child"] + spoken["examiner"])
    out["dyad_silence_frac"] = float(1.0 - any_speech / dur)
    # Overlap and interruption are omitted on purpose. The diarisation
    # assigns each moment to one speaker, so no pair of utterances in the
    # cohort overlaps (0 of 3623 adjacent pairs). Any overlap feature here
    # would be identically zero rather than a measurement.
    if any_speech > 0:
        out["dyad_child_speech_share"] = float(tot_speech["child"] / any_speech)

    for r in ROLES:
        rt = [t for t in turns if t["speaker"] == r]
        out[f"{r}_speech_frac"] = float(tot_speech[r] / dur)
        out[f"{r}_turn_rate_per_min"] = float(len(rt) / (dur / 60.0))
        if not rt:
            continue
        out.update(_desc([t["end"] - t["start"] for t in rt], f"{r}_turn_dur"))
        out.update(_desc([float(t["n_words"]) for t in rt], f"{r}_turn_words"))
        # Articulation rate excludes the silence inside a turn; speaking
        # rate does not. Bone et al. separate the two.
        art = [t["n_words"] / t["speech"] for t in rt if t["speech"] > 0.3]
        spk = [t["n_words"] / (t["end"] - t["start"]) for t in rt
               if t["end"] - t["start"] > 0.3]
        out.update(_desc(art, f"{r}_artic_rate"))
        out.update(_desc(spk, f"{r}_speak_rate"))
        pauses = [p for t in rt for p in t["pauses"]]
        out[f"{r}_intraturn_pause_per_turn"] = float(len(pauses) / len(rt))
        out.update(_desc(pauses, f"{r}_intraturn_pause"))
        held = sum(t["end"] - t["start"] for t in rt)
        if held > 0:
            out[f"{r}_intraturn_silence_frac"] = float(
                (held - sum(t["speech"] for t in rt)) / held
            )
        out[f"{r}_backchannel_frac"] = float(
            np.mean([(t["end"] - t["start"]) < BACKCHANNEL_SEC for t in rt])
        )

    # Floor transfer offset: the gap at each change of speaker. Negative
    # values are interruptions.
    fto = {r: [] for r in ROLES}
    initiated = {r: 0 for r in ROLES}
    for prev, cur in zip(turns, turns[1:]):
        if prev["speaker"] == cur["speaker"]:
            continue
        gap = cur["start"] - prev["end"]
        fto[cur["speaker"]].append(gap)
        if gap > INITIATION_SILENCE_SEC:
            initiated[cur["speaker"]] += 1
    for r in ROLES:
        g = fto[r]
        if len(g) < 3:
            continue
        out.update(_desc(g, f"{r}_fto"))
        n_resp = len(g)
        out[f"{r}_initiate_frac"] = float(initiated[r] / n_resp) if n_resp else np.nan

    # Entrainment: do the two converge on the same turn length and rate?
    ct = [t["end"] - t["start"] for t in turns if t["speaker"] == "child"]
    et = [t["end"] - t["start"] for t in turns if t["speaker"] == "examiner"]
    if len(ct) >= 4 and len(et) >= 4:
        out["dyad_turn_dur_ratio"] = float(np.median(ct) / max(1e-6, np.median(et)))
        pairs = [
            (p["end"] - p["start"], c["end"] - c["start"])
            for p, c in zip(turns, turns[1:])
            if p["speaker"] != c["speaker"]
        ]
        if len(pairs) >= 6:
            a = np.array(pairs)
            if a[:, 0].std() > 0 and a[:, 1].std() > 0:
                out["dyad_turn_dur_entrain"] = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        half = len(turns) // 2
        d1 = abs(np.median([t["end"] - t["start"] for t in turns[:half]
                            if t["speaker"] == "child"] or [np.nan])
                 - np.median([t["end"] - t["start"] for t in turns[:half]
                              if t["speaker"] == "examiner"] or [np.nan]))
        d2 = abs(np.median([t["end"] - t["start"] for t in turns[half:]
                            if t["speaker"] == "child"] or [np.nan])
                 - np.median([t["end"] - t["start"] for t in turns[half:]
                              if t["speaker"] == "examiner"] or [np.nan]))
        if np.isfinite(d1) and np.isfinite(d2):
            out["dyad_turn_convergence"] = float(d1 - d2)

    # Exchange chains: how far a back-and-forth runs before it breaks.
    chain, chains = 1, []
    for prev, cur in zip(turns, turns[1:]):
        if prev["speaker"] != cur["speaker"] and cur["start"] - prev["end"] <= 3.0:
            chain += 1
        else:
            chains.append(chain)
            chain = 1
    chains.append(chain)
    ch = np.array(chains, float)
    out["dyad_chain_mean"] = float(ch.mean())
    out["dyad_chain_max"] = float(ch.max())
    out["dyad_chain_ge4_per_min"] = float((ch >= 4).sum() / (dur / 60.0))
    return out
