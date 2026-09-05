"""Kind labels for Exp2 rank-1 features.

Used only to annotate the selected feature's content type in exp2/exp2_ridge
output CSVs (the "kind" column). win_*_delta is a measurement style, not a
kind; kind follows the content.
"""

from __future__ import annotations

KIND_SPEECH_AMOUNT = "発話の量"
KIND_SPEECH_LENGTH = "発話の長さ"
KIND_QUESTION = "質問"
KIND_BACKCHANNEL = "あいづち"
KIND_GESTURE = "身振り"
KIND_MOTION_SHARE = "動きの分担"
KIND_HEAD = "首・頭の動き"
KIND_SWAY = "揺れ"
KIND_TIMING = "やり取りの間"
KIND_COVERAGE = "映り"

_STEM_KIND: dict[str, str] = {
    "speech_frac": KIND_SPEECH_AMOUNT,
    "turn_rate_per_min": KIND_SPEECH_AMOUNT,
    "turn_words_mean": KIND_SPEECH_LENGTH,
    "backchannel_frac": KIND_BACKCHANNEL,
    "yaw_angvel_med": KIND_HEAD,
    "yaw_angacc_med": KIND_HEAD,
    "yaw_rot_per_min": KIND_HEAD,
    "roll_range": KIND_HEAD,
    "roll_angacc_mean": KIND_HEAD,
    "roll_angvel_p90": KIND_HEAD,
    "pitch_angacc_p90": KIND_HEAD,
    "pitch_reversal_rate": KIND_HEAD,
    "sway_y": KIND_SWAY,
}

_DYAD_KIND: dict[str, str] = {
    "dyad_child_speech_share": KIND_SPEECH_AMOUNT,
    "dyad_silence_frac": KIND_TIMING,
    "dyad_turn_dur_ratio": KIND_SPEECH_LENGTH,
    "dyad_turn_rate_per_min": KIND_SPEECH_AMOUNT,
    "dyad_motion_share_child": KIND_MOTION_SHARE,
    "dyad_chain_mean": KIND_TIMING,
    "dyad_chain_max": KIND_TIMING,
    "dyad_chain_ge4_per_min": KIND_TIMING,
}


def feature_kind(name: str) -> str:
    """Map one catalog column to a kind. Raises if the name is unknown."""
    if name in _DYAD_KIND:
        return _DYAD_KIND[name]
    for prefix in ("child_", "examiner_"):
        if name.startswith(prefix) and name[len(prefix) :] in _STEM_KIND:
            return _STEM_KIND[name[len(prefix) :]]
    if name.startswith("win_"):
        if "speech_frac" in name or "turn_rate" in name:
            return KIND_SPEECH_AMOUNT
        if "yawvel" in name or "yaw" in name or "roll" in name or "pitch" in name:
            return KIND_HEAD
    if name.startswith("ges_"):
        if "pose_frame_frac" in name:
            return KIND_COVERAGE
        return KIND_GESTURE
    if name.startswith("txt_"):
        if "question" in name:
            return KIND_QUESTION
        if "backchannel" in name:
            return KIND_BACKCHANNEL
        if "utt_char_cv" in name:
            return KIND_SPEECH_LENGTH
        if name.endswith("_n_utt") or "chars_per_min" in name:
            return KIND_SPEECH_AMOUNT
    if name.startswith("rsp_"):
        if name == "rsp_n_exam_turns":
            return KIND_SPEECH_AMOUNT
        return KIND_TIMING
    raise KeyError(f"no kind for feature: {name}")


def kinds_of(cols: list[str]) -> set[str]:
    return {feature_kind(c) for c in cols}


def near_far(kinds_a: set[str], kinds_b: set[str]) -> str:
    """Share ≥1 kind → 近い. Both nonempty and 0 shared → 遠い."""
    if not kinds_a or not kinds_b:
        raise ValueError("empty kind set")
    if kinds_a & kinds_b:
        return "近い"
    return "遠い"
