"""Shared data helpers for experiments 1–4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

TASKS: tuple[int, ...] = (11, 12, 3, 4, 13, 6, 1, 7, 2, 8, 9)
TASK_JA: dict[int, str] = {
    11: "誕生日パーティ",
    12: "おやつ",
    3: "ごっこ遊び",
    4: "共同で行う相互的な遊び",
    13: "ものを用いたルーティンの期待反応",
    6: "共同注意への反応",
    1: "構成課題",
    7: "実演課題",
    2: "呼名反応",
    8: "絵の叙述",
    9: "本のストーリーの説明",
}

STEMS: tuple[str, ...] = (
    "speech_frac",
    "turn_rate_per_min",
    "turn_words_mean",
    "backchannel_frac",
    "yaw_angvel_med",
    "yaw_angacc_med",
    "yaw_rot_per_min",
    "roll_range",
    "roll_angacc_mean",
    "roll_angvel_p90",
    "pitch_angacc_p90",
    "pitch_reversal_rate",
    "sway_y",
)
CHILD_COLS = [f"child_{s}" for s in STEMS]
EXAMINER_COLS = [f"examiner_{s}" for s in STEMS]
DYAD_COLS = [
    "dyad_child_speech_share",
    "dyad_silence_frac",
    "dyad_turn_dur_ratio",
    "dyad_turn_rate_per_min",
    "dyad_motion_share_child",
    "dyad_chain_mean",
    "dyad_chain_max",
    "dyad_chain_ge4_per_min",
]
FEATURE_COLS = CHILD_COLS + EXAMINER_COLS + DYAD_COLS
ALPHAS = np.logspace(-1, 4, 30)
SOURCE_OF: dict[str, str] = {
    **{c: "child" for c in CHILD_COLS},
    **{c: "examiner" for c in EXAMINER_COLS},
    **{c: "dyad" for c in DYAD_COLS},
}

TARGETS: tuple[dict[str, Any], ...] = (
    {"name": "SA", "column": "SA(対人的感情)"},
    {"name": "RRB", "column": "RRB(限定的・反復的行動)"},
    {"name": "CSS", "column": "Comparative Score"},
)

DEFAULT_FEATURES = ROOT / "outputs/features/dynamics/features_task.csv"
DEFAULT_WINDOWS = ROOT / "outputs/features/windows/features_task.csv"
DEFAULT_COHORT = ROOT / "configs/cohorts/features_63.yaml"
DEFAULT_LABELS = ROOT / "data/ADOS2_result_2.xlsx"


def load_cohort(path: Path | None = None) -> list[str]:
    cfg = yaml.safe_load((path or DEFAULT_COHORT).read_text(encoding="utf-8"))
    return [str(x) for x in cfg["participant_ids"]]


def load_labels(path: Path | None = None) -> pd.DataFrame:
    src = path or DEFAULT_LABELS
    cache = Path("/tmp/ados_labels.csv")
    if src.suffix.lower() in {".xlsx", ".xls"}:
        if (not cache.exists()) or cache.stat().st_mtime < src.stat().st_mtime:
            pd.read_excel(src, dtype=str).to_csv(cache, index=False)
        lab = pd.read_csv(cache, dtype=str)
    else:
        lab = pd.read_csv(src, dtype=str)
    lab["ID"] = lab["ID"].astype(str).str.strip().str.replace("-", "_", regex=False)
    return lab.drop_duplicates("ID").set_index("ID")


def group_of(pid: str, lab: pd.DataFrame) -> str:
    """属性 'd' → preterm, else other. Used only to stratify CV folds."""
    if pid in lab.index and "属性" in lab.columns:
        return "preterm" if str(lab.at[pid, "属性"]).strip() == "d" else "other"
    return "preterm" if str(pid).startswith("d1") else "other"


def load_features(path: Path | None = None) -> pd.DataFrame:
    """Load per-task dynamics features. No body-size correction."""
    df = pd.read_csv(path or DEFAULT_FEATURES)
    df["participant_id"] = df["participant_id"].astype(str)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns missing: {missing[:5]}...")
    return df


def apply_target_y(raw: pd.Series) -> pd.Series:
    return pd.to_numeric(raw, errors="coerce")
