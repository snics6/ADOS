"""Build participant-level tensors from per-task feature CSV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ados_ml.data.tasks import TARGET_TASK_IDS

META_COLS = {
    "participant_id",
    "task_id",
    "task_name",
    "task_duration_sec",
    "n_task_segments",
    "dyad_close_threshold",
}

# Always drop embedding placeholders when all-NaN / unused
DROP_IF_ALL_NAN = True


@dataclass
class FeatureBundle:
    participant_ids: list[str]
    task_ids: tuple[int, ...]
    feature_names: list[str]
    # (N, T, F)
    X: np.ndarray
    # labels aligned to participant_ids
    y: dict[str, np.ndarray]
    dyad_feature_indices: list[int]
    pretend_task_index: int  # index in axis=1 for task_id==3


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        if DROP_IF_ALL_NAN and df[c].isna().all():
            continue
        cols.append(c)
    return cols


def build_feature_bundle(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    task_ids: tuple[int, ...] = TARGET_TASK_IDS,
) -> FeatureBundle:
    feat_cols = _feature_columns(features_df)
    if not feat_cols:
        raise ValueError("No numeric feature columns found")

    ids = list(labels_df["participant_id"].astype(str))
    T = len(task_ids)
    F = len(feat_cols)
    X = np.full((len(ids), T, F), np.nan, dtype=np.float64)

    by_pid = {
        pid: g.set_index("task_id")
        for pid, g in features_df.groupby(features_df["participant_id"].astype(str))
    }

    for i, pid in enumerate(ids):
        if pid not in by_pid:
            raise KeyError(f"Features missing for {pid}")
        g = by_pid[pid]
        for j, tid in enumerate(task_ids):
            if tid not in g.index:
                raise KeyError(f"Task {tid} missing for {pid}")
            row = g.loc[tid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            X[i, j, :] = row[feat_cols].to_numpy(dtype=np.float64)

    y = {
        "SA": labels_df["SA"].to_numpy(dtype=np.float64),
        "RRB": labels_df["RRB"].to_numpy(dtype=np.float64),
        "C2": labels_df["C2"].to_numpy(dtype=np.float64),
        "B1": labels_df["B1_bin"].to_numpy(dtype=np.float64),
        "B12": labels_df["B12"].to_numpy(dtype=np.float64),
    }

    dyad_idx = [i for i, n in enumerate(feat_cols) if n.startswith("dyad_")]
    pretend_idx = list(task_ids).index(3)

    return FeatureBundle(
        participant_ids=ids,
        task_ids=task_ids,
        feature_names=feat_cols,
        X=X,
        y=y,
        dyad_feature_indices=dyad_idx,
        pretend_task_index=pretend_idx,
    )


def apply_ablation(
    X: np.ndarray,
    bundle: FeatureBundle,
    *,
    drop_pretend: bool = False,
    drop_dyad: bool = False,
) -> np.ndarray:
    out = X.copy()
    if drop_dyad and bundle.dyad_feature_indices:
        out[:, :, bundle.dyad_feature_indices] = 0.0
    if drop_pretend:
        out[:, bundle.pretend_task_index, :] = 0.0
    return out
