"""Confidence weights from quality fields (not used as predictors)."""

from __future__ import annotations

from ados_ml.features.stats import clip01


def pose_weight(pose: dict, *, quality_key: str = "pose_quality") -> float:
    return clip01(pose.get(quality_key)) * clip01(pose.get("role_confidence"))


def face_weight(face: dict) -> float:
    return clip01(face.get("det_score")) * clip01(face.get("role_confidence"))


def speech_weight(seg: dict) -> float:
    return clip01(seg.get("speaker_confidence"))
