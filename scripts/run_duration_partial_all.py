#!/usr/bin/env python
"""Activity-length-controlled partial correlations for every Exp1 screening hit.

For each association that survived the Exp1 BH screen, recompute Spearman
rho(x, y) and the partial rho(x, y | duration) where ``duration`` is the length
of that participant's activity interval.

The target set is read from ``outputs/exp1/hits_main.csv`` and is **not** a
fixed count: it is whatever the current Exp1 run produced. An earlier version of
this analysis hard-coded 85 associations, which silently mixed a fresh feature
table with a stale hit set.

Duration is the **union of the annotated stretches**, never the convex hull, so
the gap between two stretches of an interrupted activity is not counted as
activity time (see docs/課題区間の併合規則_2026-09-07.md). This holds for both
source tables: ``duration_sec`` in the dynamics table is ``n_frames * 0.25`` over
the union mask, and ``new_duration_sec`` in the windows table is the summed span
of the merged intervals.

    ./venv/bin/python scripts/run_duration_partial_all.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.data import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_WINDOWS,
    TARGETS,
    TASK_JA,
    apply_target_y,
    load_cohort,
    load_labels,
)
from ados_ffm.exp1_univariate import spearman_rho_fast  # noqa: E402
from ados_ffm.exp3_findings import spearman_partial  # noqa: E402

MIN_N = 8


def spearman_partial_residual(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Same quantity as ``spearman_partial``, computed by residualising.

    Rank all three, regress the ranks of x and of y on the ranks of z, then
    correlate the residuals. Algebraically identical to the closed-form partial
    formula; kept as an independent numerical check of it.
    """
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(m.sum()) < MIN_N:
        return float("nan")
    rx = rankdata(x[m], method="average")
    ry = rankdata(y[m], method="average")
    rz = rankdata(z[m], method="average")
    Z = np.column_stack([np.ones_like(rz), rz])
    try:
        ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan")
    if np.std(ex) <= 1e-12 or np.std(ey) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def _column(tab: pd.DataFrame, feat: str, tid: int) -> pd.Series:
    sub = tab[tab["task_id"].astype(int) == int(tid)]
    return pd.to_numeric(sub.set_index("participant_id")[feat], errors="coerce")


def build_table(
    hits: pd.DataFrame,
    dyn: pd.DataFrame,
    win: pd.DataFrame,
    lab: pd.DataFrame,
    cohort: list[str],
) -> pd.DataFrame:
    y_by = {
        t["name"]: apply_target_y(lab.reindex([str(p) for p in cohort])[t["column"]])
        for t in TARGETS
    }
    for y in y_by.values():
        y.index = [str(i) for i in y.index]

    rows = []
    for rec in hits.itertuples(index=False):
        feat, tid, tgt = str(rec.feature), int(rec.task), str(rec.target)
        # win_/rsp_/ges_/txt_ features live in the windows table, which carries
        # its own union duration; everything else is in the dynamics table.
        if feat in dyn.columns:
            src, dur_col = dyn, "duration_sec"
        else:
            src, dur_col = win, "new_duration_sec"
        x = _column(src, feat, tid)
        d = _column(src, dur_col, tid)
        y = y_by[tgt]
        idx = sorted(set(x.index) & set(d.index) & set(y.index))
        xv = x.reindex(idx).to_numpy(dtype=float)
        yv = y.reindex(idx).to_numpy(dtype=float)
        dv = d.reindex(idx).to_numpy(dtype=float)
        m = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(dv)
        rho = spearman_rho_fast(xv[m], yv[m])
        rho_p = spearman_partial(xv, yv, dv)
        rho_pr = spearman_partial_residual(xv, yv, dv)
        rows.append(
            {
                "feature": feat,
                "task": tid,
                "task_ja": TASK_JA[tid],
                "source": rec.source,
                "family": rec.family,
                "target": tgt,
                "n_exp1": int(rec.n),
                "n": int(m.sum()),
                "rho_exp1": float(rec.rho),
                "rho": float(rho),
                "rho_x_duration": float(spearman_rho_fast(xv[m], dv[m])),
                "rho_y_duration": float(spearman_rho_fast(yv[m], dv[m])),
                "rho_partial": float(rho_p),
                "delta": float(rho_p - rho),
                "sign_survives": bool(
                    np.isfinite(rho)
                    and np.isfinite(rho_p)
                    and np.sign(rho) == np.sign(rho_p)
                    and abs(rho_p) > 1e-9
                ),
                "rate_normalized": "per_min" in feat,
                "rho_partial_residual": float(rho_pr),
                "formula_minus_residual": float(rho_p - rho_pr),
            }
        )
    return pd.DataFrame(rows)


def flip_side_keys(signflip: pd.DataFrame) -> set[tuple[str, int, str]]:
    """(feature, task, target) for both activities of every sign-flip pair."""
    keys: set[tuple[str, int, str]] = set()
    for rec in signflip.itertuples(index=False):
        for tid in (int(rec.task_pos), int(rec.task_neg)):
            keys.add((str(rec.feature), tid, str(rec.target)))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hits", type=Path, default=ROOT / "outputs/exp1/hits_main.csv")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument("--pairs", type=Path, default=ROOT / "outputs/hetero/pairs_21.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/hetero")
    args = ap.parse_args()

    cohort = load_cohort()
    lab = load_labels()
    dyn = pd.read_csv(args.features, dtype={"participant_id": str})
    win = pd.read_csv(args.windows, dtype={"participant_id": str})
    hits = pd.read_csv(args.hits)
    hits = hits[hits["fdr_sig"].astype(bool)] if "fdr_sig" in hits else hits
    print(f"Exp1 screening hits: {len(hits)}  (from {args.hits})", flush=True)

    tab = build_table(hits, dyn, win, lab, cohort)
    out_csv = args.out / "duration_partial_all_union.csv"
    tab.to_csv(out_csv, index=False)

    n_flip = int((~tab["sign_survives"]).sum())
    ad = np.abs(tab["delta"].to_numpy(dtype=float))
    ad = ad[np.isfinite(ad)]
    worst = tab.loc[tab["delta"].abs().idxmax()]
    max_check = float(np.nanmax(np.abs(tab["formula_minus_residual"])))

    pairs = pd.read_csv(args.pairs)
    flips = pairs[pairs["sign_flip_cap"].astype(bool)]
    keys = flip_side_keys(
        flips.rename(columns={"task_a": "task_pos", "task_b": "task_neg"})
    )
    in_flip = tab.apply(
        lambda r: (r["feature"], int(r["task"]), r["target"]) in keys, axis=1
    )
    sub = tab[in_flip]

    meta = {
        "tag": "union (和集合・再計算後 2026-09-07)",
        "hits": str(args.hits),
        "n_associations": int(len(tab)),
        "n_associations_note": "read from the Exp1 output, not a fixed count",
        "features": str(args.features),
        "windows": str(args.windows),
        "duration_definition": "union of annotated stretches (gaps excluded)",
        "duration_col": {"dynamics": "duration_sec", "windows": "new_duration_sec"},
        "estimator": "ados_ffm.exp3_findings.spearman_partial (rank-Pearson partial formula)",
        "cross_check": "spearman_partial_residual (residualise ranks on rank duration)",
        "max_abs_formula_minus_residual": max_check,
        "n_sign_changes": n_flip,
        "abs_delta_median": float(np.median(ad)) if ad.size else float("nan"),
        "abs_delta_max": float(ad.max()) if ad.size else float("nan"),
        "abs_delta_argmax": {
            "feature": str(worst["feature"]),
            "task_ja": str(worst["task_ja"]),
            "target": str(worst["target"]),
        },
        "n_min": int(tab["n"].min()),
        "n_max": int(tab["n"].max()),
        "flip_sides": {
            "n_pairs": int(len(flips)),
            "n_associations": int(len(sub)),
            "n_sign_changes": int((~sub["sign_survives"]).sum()),
            "abs_delta_median": float(np.median(np.abs(sub["delta"]))) if len(sub) else float("nan"),
            "abs_delta_max": float(np.max(np.abs(sub["delta"]))) if len(sub) else float("nan"),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.out / "duration_partial_all_union_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nwrote {out_csv}  ({len(tab)} associations)")
    print(f"  符号が変わった件数      {n_flip} / {len(tab)}")
    print(f"  |Δρ| 中央値             {meta['abs_delta_median']:.4f}")
    print(f"  |Δρ| 最大               {meta['abs_delta_max']:.4f}"
          f"  ({worst['feature']} / {worst['task_ja']} / {worst['target']})")
    print(f"  n の範囲                {meta['n_min']}–{meta['n_max']}")
    print(f"  公式版 − 残差版 の最大差 {max_check:.2e}")

    print(f"\n反転 {len(flips)} 対の活動側 {len(sub)} 関連（個別）")
    cols = ["feature", "target", "task_ja", "n", "rho", "rho_partial", "delta",
            "rho_x_duration", "rho_y_duration", "sign_survives"]
    print(sub.sort_values(["feature", "target", "task"])[cols].to_string(index=False))
    print(f"\n  うち符号が変わった件数 {meta['flip_sides']['n_sign_changes']}"
          f"  |Δρ| 中央値 {meta['flip_sides']['abs_delta_median']:.4f}"
          f"  最大 {meta['flip_sides']['abs_delta_max']:.4f}")


if __name__ == "__main__":
    main()
