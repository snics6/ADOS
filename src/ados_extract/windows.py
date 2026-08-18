"""Time windows, replies, gestures, and ASR text (one row per person × task).

Speech uses diarized speaker labels. Pose/face uses vision role labels.
The two are never combined in one formula (ce_swap IDs disagree).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROLES = ("child", "examiner")
TURN_MERGE_SEC = 1.0
REPLY_SEC = 2.0
LONG_SILENCE_SEC = 2.0
BACKCHANNEL_SEC = 1.0
BACKCHANNEL_TEXTS = {"うん", "はい", "ええ", "あー", "ん", "あ", "えっ", "ふん"}
EVENT_TYPES = (
    "pointing_candidate",
    "arm_extended",
    "fidgeting",
    "hand_near_face",
    "leaning_away",
)
TASK_KEYWORDS: dict[int, tuple[str, ...]] = {
    12: ("食べ", "おいしい", "ください", "おやつ", "ジュース", "パン", "クッキー"),
    3: ("なって", "ごっこ", "だよ", "わん", "ぶー", "先生"),
    4: ("順番", "一緒", "どうぞ", "次", "番"),
}

ASR_RISKY_PREFIXES = ("txt_", "kw_")


def session_path(data_root: Path, pid: str) -> Path:
    return data_root / pid / f"{pid}_multimodal_session_v1.json"


def _role_inst(items: Any, role: str) -> dict | None:
    if not items:
        return None
    for it in items:
        if isinstance(it, dict) and it.get("role") == role:
            return it
    return None


def _merge_intervals(ivs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ivs:
        return []
    xs = sorted((float(a), float(b)) for a, b in ivs if b > a)
    out = [xs[0]]
    for a, b in xs[1:]:
        if a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _covered(t: float, ivs: list[tuple[float, float]]) -> bool:
    return any(a <= t < b for a, b in ivs)


def _progress(t: float, ivs: list[tuple[float, float]]) -> float | None:
    done = 0.0
    total = sum(b - a for a, b in ivs)
    if total <= 0:
        return None
    for a, b in ivs:
        if t < a:
            return done / total
        if t < b:
            return (done + (t - a)) / total
        done += b - a
    return 1.0


def _clip_speech(segs: Iterable[dict], ivs: list[tuple[float, float]]) -> list[dict]:
    out = []
    for s in segs:
        if s.get("speaker") not in ROLES:
            continue
        a0, b0 = float(s["start"]), float(s["end"])
        for t0, t1 in ivs:
            a, b = max(a0, t0), min(b0, t1)
            if b - a <= 0.05:
                continue
            d = dict(s)
            d["start"], d["end"] = a, b
            out.append(d)
    return sorted(out, key=lambda x: x["start"])


def _turns(utts: list[dict]) -> list[dict]:
    turns: list[dict] = []
    for u in utts:
        if (
            turns
            and turns[-1]["speaker"] == u["speaker"]
            and u["start"] - turns[-1]["end"] <= TURN_MERGE_SEC
        ):
            t = turns[-1]
            t["end"] = u["end"]
            t["texts"].append(str(u.get("text") or ""))
            t["n_utt"] += 1
            continue
        turns.append(
            {
                "speaker": u["speaker"],
                "start": u["start"],
                "end": u["end"],
                "texts": [str(u.get("text") or "")],
                "n_utt": 1,
            }
        )
    return turns


def _union_speech(utts: list[dict], role: str) -> float:
    iv = [(u["start"], u["end"]) for u in utts if u["speaker"] == role]
    return float(sum(b - a for a, b in _merge_intervals(iv)))


def _speech_window(utts: list[dict], ivs: list[tuple[float, float]], half: str) -> dict[str, float]:
    if half == "first":
        keep = [u for u in utts if (_progress(0.5 * (u["start"] + u["end"]), ivs) or 1) < 0.5]
        span = 0.5 * sum(b - a for a, b in ivs)
    else:
        keep = [u for u in utts if (_progress(0.5 * (u["start"] + u["end"]), ivs) or 0) >= 0.5]
        span = 0.5 * sum(b - a for a, b in ivs)
    out: dict[str, float] = {}
    if span <= 1.0:
        return out
    turns = _turns(keep)
    for r in ROLES:
        out[f"{r}_speech_frac"] = _union_speech(keep, r) / span
        out[f"{r}_turn_rate"] = len([t for t in turns if t["speaker"] == r]) / (span / 60.0)
    return out


def _median_abs_angvel(times: list[float], angs: list[float]) -> float:
    if len(times) < 4:
        return float("nan")
    vel = []
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if 0.1 <= dt <= 0.6 and np.isfinite(angs[i]) and np.isfinite(angs[i - 1]):
            vel.append(abs(angs[i] - angs[i - 1]) / dt)
    return float(np.median(vel)) if vel else float("nan")


def _is_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith(("？", "?", "か", "の", "かな")):
        return True
    return "？" in t or "?" in t


def _is_backchannel(text: str, dur: float) -> bool:
    t = (text or "").strip()
    if dur < BACKCHANNEL_SEC and len(t) <= 4:
        return True
    return t in BACKCHANNEL_TEXTS


def extract_one(session: dict[str, Any], task_id: int) -> dict[str, float] | None:
    segs = [
        t
        for t in session.get("task_segments") or []
        if int(t.get("task_id", -1)) == int(task_id)
    ]
    ivs = _merge_intervals(
        [
            (float(t["session_start_sec"]), float(t["session_end_sec"]))
            for t in segs
            if t.get("session_start_sec") is not None and t.get("session_end_sec") is not None
        ]
    )
    if not ivs:
        return None
    dur = sum(b - a for a, b in ivs)
    if dur < 20.0:
        return None
    minutes = dur / 60.0
    speech = _clip_speech(session.get("speech_segments") or [], ivs)
    turns = _turns(speech)
    row: dict[str, float] = {
        "new_duration_sec": float(dur),
        "new_n_task_spans": float(len(ivs)),
    }

    # --- time windows (first vs second half of covered time) ---
    first = _speech_window(speech, ivs, "first")
    second = _speech_window(speech, ivs, "second")
    for r in ROLES:
        a = first.get(f"{r}_speech_frac", np.nan)
        b = second.get(f"{r}_speech_frac", np.nan)
        row[f"win_{r}_speech_frac_delta"] = float(b - a) if np.isfinite(a) and np.isfinite(b) else np.nan
        a = first.get(f"{r}_turn_rate", np.nan)
        b = second.get(f"{r}_turn_rate", np.nan)
        row[f"win_{r}_turn_rate_delta"] = float(b - a) if np.isfinite(a) and np.isfinite(b) else np.nan

    yaw_t = {r: {h: [] for h in ("first", "second")} for r in ROLES}
    yaw_v = {r: {h: [] for h in ("first", "second")} for r in ROLES}
    ev_counts = {r: {e: 0 for e in EVENT_TYPES} for r in ROLES}
    pose_frames = {r: 0 for r in ROLES}

    for fr in session.get("timeline") or []:
        t = float(fr.get("time_sec", np.nan))
        if not np.isfinite(t) or not _covered(t, ivs):
            continue
        prog = _progress(t, ivs)
        half = "first" if (prog or 1) < 0.5 else "second"
        for r in ROLES:
            face = _role_inst(fr.get("face"), r)
            hp = (face or {}).get("head_pose") or {}
            yaw = hp.get("yaw")
            if yaw is not None and np.isfinite(float(yaw)):
                yaw_t[r][half].append(t)
                yaw_v[r][half].append(float(yaw))
            pose = _role_inst(fr.get("pose"), r)
            if pose is None:
                continue
            pose_frames[r] += 1
            types = {ev.get("type") for ev in (pose.get("events") or []) if isinstance(ev, dict)}
            for e in EVENT_TYPES:
                if e in types:
                    ev_counts[r][e] += 1

    for r in ROLES:
        v1 = _median_abs_angvel(yaw_t[r]["first"], yaw_v[r]["first"])
        v2 = _median_abs_angvel(yaw_t[r]["second"], yaw_v[r]["second"])
        row[f"win_{r}_yawvel_delta"] = (
            float(v2 - v1) if np.isfinite(v1) and np.isfinite(v2) else np.nan
        )
        for e in EVENT_TYPES:
            row[f"ges_{r}_{e}_per_min"] = float(ev_counts[r][e] / minutes)
        row[f"ges_{r}_pose_frame_frac"] = float(pose_frames[r] / max(dur / 0.25, 1.0))

    # --- response structure (speech only) ---
    child_replies = 0
    exam_turns = 0
    gaps: list[float] = []
    for prev, cur in zip(turns, turns[1:]):
        if prev["speaker"] != "examiner":
            continue
        exam_turns += 1
        if cur["speaker"] != "child":
            continue
        gap = cur["start"] - prev["end"]
        gaps.append(gap)
        if 0.0 <= gap <= REPLY_SEC:
            child_replies += 1
    row["rsp_child_reply_frac"] = (
        float(child_replies / exam_turns) if exam_turns else np.nan
    )
    if gaps:
        ga = np.asarray(gaps, dtype=float)
        row["rsp_after_exam_gap_med"] = float(np.median(ga))
        row["rsp_after_exam_gap_p90"] = float(np.percentile(ga, 90))
        row["rsp_long_silence_frac"] = float(np.mean(ga > LONG_SILENCE_SEC))
    else:
        row["rsp_after_exam_gap_med"] = np.nan
        row["rsp_after_exam_gap_p90"] = np.nan
        row["rsp_long_silence_frac"] = np.nan
    row["rsp_n_exam_turns"] = float(exam_turns)

    # --- exploratory text (ASR-risky where noted) ---
    for r in ROLES:
        utt = [u for u in speech if u["speaker"] == r]
        texts = [str(u.get("text") or "") for u in utt]
        durs = [float(u["end"] - u["start"]) for u in utt]
        n = len(texts)
        chars = np.array([len(t) for t in texts], dtype=float) if n else np.array([])
        row[f"txt_{r}_n_utt"] = float(n)
        row[f"txt_{r}_question_frac"] = (
            float(np.mean([_is_question(t) for t in texts])) if n else np.nan
        )
        row[f"txt_{r}_backchannel_frac"] = (
            float(np.mean([_is_backchannel(t, d) for t, d in zip(texts, durs)]))
            if n
            else np.nan
        )
        if n >= 3 and float(np.mean(chars)) > 0:
            row[f"txt_{r}_utt_char_cv"] = float(np.std(chars) / np.mean(chars))
        else:
            row[f"txt_{r}_utt_char_cv"] = np.nan
        row[f"txt_{r}_chars_per_min"] = float(chars.sum() / minutes) if n else np.nan

    # --- task-specific keywords (ASR-risky) ---
    kws = TASK_KEYWORDS.get(int(task_id), ())
    blob_c = "".join(str(u.get("text") or "") for u in speech if u["speaker"] == "child")
    blob_e = "".join(str(u.get("text") or "") for u in speech if u["speaker"] == "examiner")
    for kw in kws:
        row[f"kw_child_{kw}_per_min"] = float(blob_c.count(kw) / minutes)
        row[f"kw_exam_{kw}_per_min"] = float(blob_e.count(kw) / minutes)
    return row


def extract_windows(
    data_root: Path,
    pids: list[str],
    task_ids: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for pid in pids:
        path = session_path(data_root, pid)
        if not path.exists():
            continue
        session = json.loads(path.read_text(encoding="utf-8"))
        for tid in task_ids:
            feat = extract_one(session, tid)
            if feat is None:
                continue
            feat["participant_id"] = str(pid)
            feat["task_id"] = int(tid)
            rows.append(feat)
    if not rows:
        return pd.DataFrame(columns=["participant_id", "task_id"])
    return pd.DataFrame(rows)


def feature_family(name: str) -> str:
    if name.startswith("win_"):
        return "new_window"
    if name.startswith("rsp_"):
        return "new_response"
    if name.startswith("ges_"):
        return "new_gesture"
    if name.startswith("txt_"):
        return "new_text"
    if name.startswith("kw_"):
        return "new_keyword"
    return "new_other"


def is_asr_risky(name: str) -> bool:
    return name.startswith(ASR_RISKY_PREFIXES)
