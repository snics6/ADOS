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

A task interval is a **set of disjoint spans**, not one stretch: an activity
interrupted and resumed is annotated as several stretches, and the time in
between belongs to whatever ran instead. Every function here therefore takes
``spans`` rather than a ``(t0, t1)`` pair, and:

* rates are divided by the union of the spans, never by the enclosing hull;
* turns, exchange chains and floor-transfer offsets are formed **inside one
  span only**. A silence that contains another activity is not a response
  latency, so no pair of turns is allowed to straddle a span boundary.
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


Span = tuple[float, float]


def normalise_spans(spans: Sequence[Span] | Span) -> list[Span]:
    """Accept one (t0, t1) pair or a sequence of them; return disjoint spans."""
    if (
        len(spans) == 2
        and not isinstance(spans[0], (tuple, list))
        and not isinstance(spans[1], (tuple, list))
    ):
        spans = [(float(spans[0]), float(spans[1]))]  # type: ignore[assignment]
    xs = sorted((float(a), float(b)) for a, b in spans if float(b) > float(a))
    if not xs:
        return []
    out = [xs[0]]
    for a, b in xs[1:]:
        if a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def span_total(spans: Sequence[Span]) -> float:
    return float(sum(b - a for a, b in spans))


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
    speech_segments: Sequence[dict[str, Any]], spans: Sequence[Span] | Span
) -> dict[str, float]:
    """Turn-taking timing for both speakers over a set of disjoint spans."""
    out: dict[str, float] = {}
    spans = normalise_spans(spans)
    # Rates are per minute of activity, so the denominator is the union of the
    # spans. The hull would count the interruption as activity time.
    dur = span_total(spans)
    if dur <= 1.0:
        return out
    # Turns are built per span, so no turn merges across an interruption and
    # no pair of turns straddles one.
    turn_groups = [
        _turns([u for u in _clip(speech_segments, a, b) if u.get("speaker") in ROLES])
        for a, b in spans
    ]
    utts = [u for a, b in spans
            for u in _clip(speech_segments, a, b) if u.get("speaker") in ROLES]
    if len(utts) < 4:
        return out
    turns = [t for g in turn_groups for t in g]
    # Consecutive turn pairs *within* a span. Cross-span pairs are dropped:
    # the silence between two stretches contains another activity, so it is
    # neither a response latency nor a chance to continue a chain.
    pairs = [(a, b) for g in turn_groups for a, b in zip(g, g[1:])]
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
    for prev, cur in pairs:
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
        adj = [
            (p["end"] - p["start"], c["end"] - c["start"])
            for p, c in pairs
            if p["speaker"] != c["speaker"]
        ]
        if len(adj) >= 6:
            a = np.array(adj)
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
    out.update(chain_initiator_features_from_turns(turn_groups, dur))
    return out


CHAIN_GAP_SEC = 3.0


def _exchange_chains(turn_groups: Sequence[list[dict]]) -> tuple[np.ndarray, list[str]]:
    """Lengths and initiator (first speaker) of each exchange chain.

    A chain is a run of alternating turns with no gap longer than
    ``CHAIN_GAP_SEC``. Chains are counted inside one span at a time, so a
    stretch boundary always ends the chain in progress.
    """
    chains: list[int] = []
    inits: list[str] = []
    for turns in turn_groups:
        if not turns:
            continue
        chain = 1
        current_init = str(turns[0]["speaker"])
        for prev, cur in zip(turns, turns[1:]):
            if (
                prev["speaker"] != cur["speaker"]
                and cur["start"] - prev["end"] <= CHAIN_GAP_SEC
            ):
                chain += 1
                continue
            chains.append(chain)
            inits.append(current_init)
            chain = 1
            current_init = str(cur["speaker"])
        chains.append(chain)
        inits.append(current_init)
    return np.asarray(chains, float), inits


def chain_initiator_features_from_turns(
    turn_groups: Sequence[list[dict]], dur: float
) -> dict[str, float]:
    """Overall chain stats plus a split by who started the chain."""
    out: dict[str, float] = {}
    ch, inits = _exchange_chains(turn_groups)
    if ch.size == 0 or dur <= 1.0:
        return out
    out["dyad_chain_mean"] = float(ch.mean())
    out["dyad_chain_max"] = float(ch.max())
    out["dyad_chain_ge4_per_min"] = float((ch >= 4).sum() / (dur / 60.0))
    out["dyad_chain_n"] = float(ch.size)
    for role, key in (("examiner", "exam"), ("child", "child")):
        mask = np.array([s == role for s in inits], dtype=bool)
        sub = ch[mask]
        out[f"dyad_chain_n_{key}_init"] = float(sub.size)
        if sub.size == 0:
            out[f"dyad_chain_mean_{key}_init"] = float("nan")
            out[f"dyad_chain_max_{key}_init"] = float("nan")
            out[f"dyad_chain_ge4_per_min_{key}_init"] = 0.0
            continue
        out[f"dyad_chain_mean_{key}_init"] = float(sub.mean())
        out[f"dyad_chain_max_{key}_init"] = float(sub.max())
        out[f"dyad_chain_ge4_per_min_{key}_init"] = float((sub >= 4).sum() / (dur / 60.0))
    return out


def chain_initiator_features(
    speech_segments: Sequence[dict[str, Any]], spans: Sequence[Span] | Span
) -> dict[str, float]:
    """Chain stats from speech only (no pose). Used for the initiator split."""
    spans = normalise_spans(spans)
    dur = span_total(spans)
    if dur <= 1.0:
        return {}
    turn_groups = [
        _turns([u for u in _clip(speech_segments, a, b) if u.get("speaker") in ROLES])
        for a, b in spans
    ]
    if sum(len(g) for g in turn_groups) < 1:
        return {}
    n_utt = sum(
        len([u for u in _clip(speech_segments, a, b) if u.get("speaker") in ROLES])
        for a, b in spans
    )
    if n_utt < 4:
        return {}
    return chain_initiator_features_from_turns(turn_groups, dur)
