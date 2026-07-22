"""Participant ID normalization for labels ↔ feature folders."""

from __future__ import annotations


def normalize_participant_id(raw: str | int | float) -> str:
    """Map label-sheet IDs to feature-folder IDs.

    Examples:
        020401 -> 020401
        d1-402 -> d1_402
        20401 (int) -> 020401  (zero-pad to 6 if purely numeric and len < 6)
    """
    if isinstance(raw, float):
        if raw.is_integer():
            raw = int(raw)
        else:
            raise ValueError(f"Non-integer ID: {raw}")
    s = str(raw).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    s = s.replace("-", "_")
    if s.isdigit() and len(s) < 6:
        s = s.zfill(6)
    return s
