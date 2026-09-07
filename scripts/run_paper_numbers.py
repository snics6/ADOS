"""Numbers the paper body needs, recomputed from the current outputs.

Five blocks, all read from outputs/ (nothing is re-fit here except the
rank-average re-rank comparison, which needs the feature tables):

  1. Fisher-z decomposition of the cross-task correlations (r_all_pairs.csv)
  2. participants per task pair
  3. cross-task agreement by feature family / source / kind
  4. Eq.3 check and the "re-rank the rank average or not" gap
  5. the positive-side single-activity CI for one Table 3 row
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.data import TARGETS, apply_target_y, load_cohort, load_labels  # noqa: E402
from ados_ffm.exp1_univariate import spearman_rho_fast  # noqa: E402
from ados_ffm.kinds import feature_kind  # noqa: E402

HETERO = ROOT / "outputs/hetero"
WITHIN_CHANGE_FAMILY = "new_window"  # the six win_*_delta features


def fisher_z_decomposition(r: pd.DataFrame) -> dict:
    d = r.dropna(subset=["r", "n"])
    d = d[d["n"] > 3]
    z = np.arctanh(d["r"].clip(-0.999999, 0.999999).to_numpy(dtype=float))
    var_obs = float(np.var(z, ddof=1))
    var_samp = float(np.mean(1.0 / (d["n"].to_numpy(dtype=float) - 3.0)))
    return {
        "n_correlations": int(len(d)),
        "var_observed": var_obs,
        "var_sampling_mean": var_samp,
        "sampling_share": var_samp / var_obs,
        "sd_true": float(np.sqrt(max(0.0, var_obs - var_samp))),
        "sd_observed": float(np.sqrt(var_obs)),
    }


def pair_n(r: pd.DataFrame) -> dict:
    per_pair_max = r.groupby(["task_a", "task_b"])["n"].max()
    return {
        "per_row": {
            "count": int(len(r)),
            "min": int(r["n"].min()),
            "max": int(r["n"].max()),
            "median": float(r["n"].median()),
        },
        "per_pair_max": {
            "count": int(len(per_pair_max)),
            "min": int(per_pair_max.min()),
            "max": int(per_pair_max.max()),
            "median": float(per_pair_max.median()),
        },
    }


def agreement(r: pd.DataFrame) -> dict:
    r = r.copy()
    r["kind"] = r["feature"].map(feature_kind)
    within = r[r["family"] == WITHIN_CHANGE_FAMILY]
    fam = r.groupby("family")["r"].median()
    other = fam.drop(WITHIN_CHANGE_FAMILY)
    src = r.groupby("source")["r"].median()
    per_feat = r.groupby("feature")["r"].median().sort_values(ascending=False)
    return {
        "within_activity_change": {
            "n_features": int(within["feature"].nunique()),
            "n_correlations": int(len(within)),
            "median_r": float(within["r"].median()),
            "frac_negative": float((within["r"] < 0).mean()),
        },
        "family_median": {k: float(v) for k, v in fam.items()},
        "other_families_range": [float(other.min()), float(other.max())],
        "source_median": {k: float(v) for k, v in src.items()},
        "source_range": [float(src.min()), float(src.max())],
        "kind_median": {
            k: float(v) for k, v in r.groupby("kind")["r"].median().items()
        },
        "top_features": {k: float(v) for k, v in per_feat.head(8).items()},
        "top5_range": [float(per_feat.head(5).min()), float(per_feat.head(5).max())],
    }


def _feature_column(
    dyn: pd.DataFrame, win: pd.DataFrame, feat: str, tid: int
) -> pd.Series:
    tab = win if feat in win.columns else dyn
    sub = tab[tab["task_id"].astype(int) == int(tid)]
    return pd.to_numeric(sub.set_index("participant_id")[feat], errors="coerce")


def eq3_and_rerank(pv: pd.DataFrame) -> dict:
    diff = (pv["rho_z_pred"] - pv["rho_z_obs"]).abs()
    dyn = pd.read_csv(ROOT / "outputs/features/dynamics/features_task.csv")
    win = pd.read_csv(ROOT / "outputs/features/windows/features_task.csv")
    for tab in (dyn, win):
        tab["participant_id"] = tab["participant_id"].astype(str)
    cohort = load_cohort()
    lab = load_labels()
    y_by = {}
    for tgt in TARGETS:
        s = apply_target_y(lab.reindex([str(p) for p in cohort])[tgt["column"]])
        s.index = [str(i) for i in s.index]
        y_by[tgt["name"]] = s

    rows = []
    for rec in pv.itertuples(index=False):
        xa = _feature_column(dyn, win, rec.feature, int(rec.task_pos))
        xb = _feature_column(dyn, win, rec.feature, int(rec.task_neg))
        y = y_by[str(rec.target)]
        idx = sorted(set(xa.index) & set(xb.index) & set(y.index))
        a, b, yy = (s.reindex(idx).to_numpy(dtype=float) for s in (xa, xb, y))
        m = np.isfinite(a) & np.isfinite(b) & np.isfinite(yy)
        a, b, yy = a[m], b[m], yy[m]
        z = 0.5 * (rankdata(a) + rankdata(b))
        ry = rankdata(yy)
        zc, ryc = z - z.mean(), ry - ry.mean()
        no_rerank = float(zc @ ryc / np.sqrt((zc @ zc) * (ryc @ ryc)))
        rerank = spearman_rho_fast(z, yy)
        rows.append(
            {
                "feature": rec.feature,
                "target": rec.target,
                "task_pos": int(rec.task_pos),
                "task_neg": int(rec.task_neg),
                "n": int(m.sum()),
                "rho_z_no_rerank": no_rerank,
                "rho_z_rerank": rerank,
                "diff": rerank - no_rerank,
            }
        )
    t = pd.DataFrame(rows)
    ad = t["diff"].abs()
    return {
        "eq3": {
            "n_pairs": int(len(pv)),
            "abs_err_median": float(diff.median()),
            "abs_err_max": float(diff.max()),
            "corr_pred_obs_pearson": float(
                np.corrcoef(pv["rho_z_pred"], pv["rho_z_obs"])[0, 1]
            ),
            "n_boot": sorted(set(int(v) for v in pv["n_boot"])),
        },
        "rerank_vs_not": {
            "definition": "current pipeline uses the rank average as-is (P1); the alternative re-ranks z",
            "abs_diff_median": float(ad.median()),
            "abs_diff_max": float(ad.max()),
            "corr": float(np.corrcoef(t["rho_z_no_rerank"], t["rho_z_rerank"])[0, 1]),
            "argmax": t.loc[ad.idxmax(), ["feature", "target", "n"]].to_dict(),
        },
        "_table": t,
    }


def table3_row(pv: pd.DataFrame, feature: str, target: str) -> dict:
    s = pv[(pv["feature"] == feature) & (pv["target"] == target)]
    if s.empty:
        return {"found": False, "feature": feature, "target": target}
    r = s.iloc[0]
    return {
        "found": True,
        "feature": feature,
        "target": target,
        "positive_side_task": int(r["task_pos"]),
        "positive_side_task_ja": str(r["task_pos_ja"]),
        "negative_side_task_ja": str(r["task_neg_ja"]),
        "n": int(r["n_cap"]),
        "rho": float(r["rho_pos"]),
        "ci95": [float(r["rho_pos_lo"]), float(r["rho_pos_hi"])],
        "excludes_zero": bool(r["rho_pos_excludes_0"]),
        "ci_method": "subject bootstrap, percentile, ranks recomputed inside each replicate",
        "n_boot": int(r["n_boot"]),
        "sample": "intersection of the two activities (n_cap)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HETERO / "paper_numbers.json"))
    ap.add_argument("--feature", default="examiner_turn_rate_per_min")
    ap.add_argument("--target", default="SA")
    args = ap.parse_args()

    r = pd.read_csv(HETERO / "r_all_pairs.csv")
    pv = pd.read_csv(HETERO / "pool_verify_10.csv")

    eq3 = eq3_and_rerank(pv)
    rerank_table = eq3.pop("_table")
    out = {
        "1_fisher_z": fisher_z_decomposition(r),
        "2_pair_n": pair_n(r),
        "3_agreement": agreement(r),
        "4_eq3_and_rerank": eq3,
        "5_table3_positive_side": table3_row(pv, args.feature, args.target),
        "sources": {
            "r_all_pairs": str(HETERO / "r_all_pairs.csv"),
            "pool_verify": str(HETERO / "pool_verify_10.csv"),
            "features": str(ROOT / "outputs/features"),
        },
    }
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rerank_table.to_csv(HETERO / "rerank_check_10.csv", index=False)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
