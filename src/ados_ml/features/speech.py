"""Speech child / dyad features for one task window set."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ados_ml.data.tasks import Interval
from ados_ml.features.confidence import speech_weight
from ados_ml.features.stats import safe_div, weighted_mean


EmbedFn = Callable[[list[str]], np.ndarray]  # (n, d)


def _overlap(seg_start: float, seg_end: float, intervals: list[Interval]) -> float:
    return float(sum(iv.overlap(seg_start, seg_end) for iv in intervals))


def extract_speech_features(
    speech_segments: Sequence[dict],
    intervals: list[Interval],
    task_dur: float,
    *,
    tau_sp: float = 0.3,
    embed_fn: EmbedFn | None = None,
    max_response_gap: float = 10.0,
) -> dict[str, Any]:
    if task_dur <= 0 or not intervals:
        return _empty_speech(embed_dim=None if embed_fn is None else 0)

    # Collect overlapping segments
    rows: list[dict[str, Any]] = []
    for seg in speech_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        ov = _overlap(start, end, intervals)
        if ov <= 0:
            continue
        w = speech_weight(seg)
        if w < tau_sp:
            continue
        speaker = seg.get("speaker")
        if speaker not in ("child", "examiner", "other"):
            speaker = "other"
        rows.append(
            {
                "start": start,
                "end": end,
                "overlap": ov,
                "speaker": speaker,
                "weight": w,
                "text": (seg.get("text") or "").strip(),
            }
        )

    rows.sort(key=lambda r: (r["start"], r["end"]))

    child_rows = [r for r in rows if r["speaker"] == "child"]
    ex_rows = [r for r in rows if r["speaker"] == "examiner"]

    child_time = float(sum(r["overlap"] for r in child_rows))
    ex_time = float(sum(r["overlap"] for r in ex_rows))

    # Overlap time via 0.25 s grid on the union of task intervals
    speech_overlap = 0.0
    step = 0.25
    for iv in intervals:
        t = iv.start
        while t < iv.end:
            child_on = any(r["start"] <= t < r["end"] for r in child_rows)
            ex_on = any(r["start"] <= t < r["end"] for r in ex_rows)
            if child_on and ex_on:
                speech_overlap += step
            t += step
    speech_overlap = min(speech_overlap, task_dur)

    turn_count = 0
    for i in range(1, len(rows)):
        if rows[i]["speaker"] != rows[i - 1]["speaker"]:
            if rows[i]["speaker"] in ("child", "examiner") and rows[i - 1]["speaker"] in (
                "child",
                "examiner",
            ):
                turn_count += 1

    latencies: list[float] = []
    lat_w: list[float] = []
    follow = 0
    for i, r in enumerate(rows):
        if r["speaker"] != "child":
            continue
        # previous examiner
        prev_ex = None
        for j in range(i - 1, -1, -1):
            if rows[j]["speaker"] == "examiner":
                prev_ex = rows[j]
                break
            if rows[j]["speaker"] == "child":
                break
        if prev_ex is None:
            continue
        follow += 1
        gap = r["start"] - prev_ex["end"]
        if gap < 0:
            gap = 0.0
        if gap > max_response_gap:
            continue
        latencies.append(gap)
        lat_w.append(r["weight"])

    child_chars = float(sum(len(r["text"]) for r in child_rows))
    utt_durs = [r["overlap"] for r in child_rows]
    utt_w = [r["weight"] for r in child_rows]

    out: dict[str, Any] = {
        "child_speech_time_frac": safe_div(child_time, task_dur),
        "child_n_utterances": float(len(child_rows)),
        "child_utt_dur_wmean": weighted_mean(utt_durs, utt_w),
        "child_chars_per_sec": safe_div(child_chars, task_dur),
        "dyad_turn_count": float(turn_count),
        "dyad_turn_rate": safe_div(float(turn_count), task_dur),
        "child_response_latency_wmean": weighted_mean(latencies, lat_w),
        "examiner_to_child_frac": safe_div(float(follow), float(len(child_rows)))
        if child_rows
        else float("nan"),
        "speech_overlap_frac": safe_div(speech_overlap, task_dur),
        "child_vs_examiner_time_ratio": safe_div(child_time, ex_time),
    }

    # Embeddings
    if embed_fn is None:
        out["child_emb_disp"] = float("nan")
        out["dyad_emb_diff_norm"] = float("nan")
        out["child_emb_mean"] = None
        return out

    child_texts = [r["text"] for r in child_rows if r["text"]]
    ex_texts = [r["text"] for r in ex_rows if r["text"]]
    if child_texts:
        emb_c = embed_fn(child_texts)
        # L2 normalize rows
        norms = np.linalg.norm(emb_c, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_c = emb_c / norms
        mean_c = emb_c.mean(axis=0)
        disp = float(emb_c.std(axis=0).mean()) if len(emb_c) > 1 else 0.0
    else:
        mean_c = None
        disp = float("nan")

    if ex_texts:
        emb_e = embed_fn(ex_texts)
        norms = np.linalg.norm(emb_e, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_e = emb_e / norms
        mean_e = emb_e.mean(axis=0)
    else:
        mean_e = None

    if mean_c is not None and mean_e is not None:
        diff = float(np.linalg.norm(mean_c - mean_e))
    else:
        diff = float("nan")

    out["child_emb_mean"] = None if mean_c is None else mean_c.astype(np.float32)
    out["child_emb_disp"] = disp
    out["dyad_emb_diff_norm"] = diff
    return out


def _empty_speech(*, embed_dim: int | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "child_speech_time_frac": float("nan"),
        "child_n_utterances": 0.0,
        "child_utt_dur_wmean": float("nan"),
        "child_chars_per_sec": float("nan"),
        "dyad_turn_count": 0.0,
        "dyad_turn_rate": float("nan"),
        "child_response_latency_wmean": float("nan"),
        "examiner_to_child_frac": float("nan"),
        "speech_overlap_frac": float("nan"),
        "child_vs_examiner_time_ratio": float("nan"),
        "child_emb_disp": float("nan"),
        "dyad_emb_diff_norm": float("nan"),
        "child_emb_mean": None,
    }
    return out
