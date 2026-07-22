"""Cohort YAML helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ados_ml.utils.config import load_yaml
from ados_ml.utils.ids import normalize_participant_id


def load_cohort(path: Path) -> dict[str, Any]:
    raw = load_yaml(path)
    ids = [normalize_participant_id(x) for x in raw.get("participant_ids", [])]
    raw["participant_ids"] = ids
    raw["ce_swap_ids"] = [
        normalize_participant_id(x) for x in raw.get("ce_swap_ids", [])
    ]
    return raw
