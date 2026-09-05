#!/usr/bin/env python
"""Regenerate Experiment 1 hit scatter plots (English, no regression line).

Uses existing outputs/exp1/hits_main.csv and the feature tables.

    ./venv/bin/python -u scripts/plot_exp1_hits.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.data import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_WINDOWS,
    TARGETS,
    load_cohort,
    load_features,
    load_labels,
)
from ados_ffm.exp1_univariate import (  # noqa: E402
    catalog_cols,
    feature_payloads,
    merge_task_catalog,
    y_map_for_task,
)

# English ADOS Module 2 task names (avoid CJK fonts).
TASK_EN: dict[int, str] = {
    1: "Construction",
    2: "Response to Name",
    3: "Pretend+Joint Play",
    4: "Joint Interactive Play",
    6: "Response to Joint Attention",
    7: "Demonstration",
    8: "Picture+Book",
    9: "Telling a Story from a Book",
    11: "Birthday+Snack",
    12: "Snack",
    13: "Routine with Objects",
}


def _safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "x"


def _payload_map(
    dyn: pd.DataFrame,
    win: pd.DataFrame,
    lab: pd.DataFrame,
    cohort: list[str],
) -> dict[tuple[int, str, str], dict]:
    """(task, target, feature) -> payload with x, y, n."""
    out: dict[tuple[int, str, str], dict] = {}
    tgt_by_name = {t["name"]: t for t in TARGETS}
    tasks = sorted({int(t) for t in dyn["task_id"].dropna().unique()})
    for tid in tasks:
        frame = merge_task_catalog(dyn, win, tid, cohort)
        if frame.empty:
            continue
        cols = catalog_cols(list(frame.columns))
        for tname, tgt in tgt_by_name.items():
            ymap = y_map_for_task(frame, lab, tgt)
            if len(ymap) < 8:
                continue
            for pl in feature_payloads(frame, ymap, cols):
                out[(tid, tname, pl["feature"])] = pl
    return out


def _plot_one(row: pd.Series, pl: dict, out_path: Path) -> None:
    x = np.asarray(pl["x"], dtype=float)
    y = np.asarray(pl["y"], dtype=float)
    tid = int(row["task"])
    task_en = TASK_EN.get(tid, f"Activity {tid}")
    feature = str(row["feature"])
    target = str(row["target"])
    source = str(row["source"])
    n = int(row["n"]) if pd.notna(row["n"]) else int(pl["n"])
    rho = float(row["rho"])
    q = float(row["q_fdr"])

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(x, y, s=28, alpha=0.75, edgecolors="none", color="#1f4e79")
    ax.set_title(
        f"Activity {tid}: {task_en}  |  {target}\n{feature}  ({source})",
        fontsize=10,
    )
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    note = f"n={n}   ρ={rho:.3f}   q={q:.4g}"
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#cccccc",
        },
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hits", type=Path, default=ROOT / "outputs/exp1/hits_main.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/exp1/plots")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    args = ap.parse_args()

    if not args.hits.exists():
        raise SystemExit(f"hits not found: {args.hits}")
    hits = pd.read_csv(args.hits)
    if hits.empty:
        raise SystemExit("hits table is empty")

    cohort = load_cohort()
    lab = load_labels()
    dyn = load_features(path=args.features)
    dyn = dyn[dyn["participant_id"].astype(str).isin(cohort)].copy()
    win = pd.read_csv(args.windows)
    win["participant_id"] = win["participant_id"].astype(str)

    print(f"plot_exp1_hits  hits={len(hits)}  ->  {args.out}", flush=True)
    payloads = _payload_map(dyn, win, lab, cohort)
    print(f"payloads  {len(payloads)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.png"):
        old.unlink()

    n_ok = 0
    n_miss = 0
    for _, row in hits.iterrows():
        key = (int(row["task"]), str(row["target"]), str(row["feature"]))
        pl = payloads.get(key)
        if pl is None:
            n_miss += 1
            print(f"miss  {key}", flush=True)
            continue
        fname = (
            f"t{int(row['task'])}_{row['target']}_{row['source']}_"
            f"{_safe_name(row['feature'])}.png"
        )
        _plot_one(row, pl, args.out / fname)
        n_ok += 1

    print(f"done  written={n_ok}  missing={n_miss}  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
