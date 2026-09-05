#!/usr/bin/env python
"""Experiment 3: 実験1で複数課題に出た21対の相関の差.

The 21 (feature, target) pairs that showed up in two or more Exp1 FDR hits.
Pair tests (H0: ρ_A = ρ_B) and Eq. 3 identity checks are auxiliary. The
data claims live in `scripts/run_exp3_findings.py`.

    ./venv/bin/python -u scripts/run_hetero.py
    ./venv/bin/python -u scripts/run_hetero.py --self-check
    ./venv/bin/python -u scripts/run_hetero.py --q-supp
    ./venv/bin/python -u scripts/run_exp3_findings.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.data import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_WINDOWS,
    TASK_JA,
    TASKS,
    TARGETS,
    load_cohort,
    load_features,
    load_labels,
)
from ados_ffm.exp1_univariate import spearman_rho_fast  # noqa: E402
from ados_ffm.hetero import (  # noqa: E402
    N_BOOT_PAIR,
    N_BOOT_Q,
    N_BOOT_VERIFY,
    SEED_PAIR,
    SEED_Q,
    attach_bh_global,
    attach_bh_one,
    bootstrap_q,
    build_panel,
    exp1_multitask_pairs,
    family_seed,
    feature_meta,
    observed_tasks,
    pair_contrast,
    pick_figure_pair,
    plot_eq3_verify,
    pool_verify_summary,
    self_check,
    verify_signflip_table,
)

HIT_DEFAULT = ROOT / "outputs/exp1/hits_main.csv"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


MARKERS = {"child": "o", "examiner": "s", "dyad": "D"}
COLORS = {"SA": "#c0392b", "RRB": "#2980b9", "CSS": "#1e8449"}


def attach_pool_gains(pairs: pd.DataFrame) -> pd.DataFrame:
    """Δρ vs the stronger task, after the same orientation as λ.

    gain = ρ_z − ρ_A  (ρ_A > 0). Positive = pool better. Eq. 3 gives the prediction.
    """
    out = pairs.copy()
    cap_a = out["rho_cap_a"].to_numpy(dtype=float)
    sgn = np.where(np.isfinite(cap_a) & (cap_a < 0.0), -1.0, 1.0)
    a = out["rho_a_orient"].to_numpy(dtype=float)
    out["p1_gain_pred"] = sgn * out["p1_rho_z_pred"].to_numpy(dtype=float) - a
    out["p1_gain_obs"] = sgn * out["p1_rho_z_obs"].to_numpy(dtype=float) - a
    return out


def _scatter_pairs(ax, df: pd.DataFrame, xcol: str, ycol: str, *, size, legend: bool) -> None:
    for tgt, g in df.groupby("target"):
        for src, gg in g.groupby("source"):
            s = size(gg) if callable(size) else size
            ax.scatter(
                gg[xcol],
                gg[ycol],
                s=s,
                c=COLORS.get(str(tgt), "#555555"),
                marker=MARKERS.get(str(src), "o"),
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
                label=f"{tgt} / {src}" if legend else None,
            )


def plot_lambda_r(pairs: pd.DataFrame, path: Path) -> None:
    from ados_ffm.exp3_findings import plot_lambda_r_data

    plot_lambda_r_data(pairs, path)


def _contrast_row(panel, feat: str, tgt: str, ta: int, tb: int, n_boot: int) -> dict:
    ta, tb = int(ta), int(tb)
    lo, hi = (ta, tb) if ta <= tb else (tb, ta)
    got = pair_contrast(
        panel.X[feat][ta],
        panel.X[feat][tb],
        panel.y[tgt],
        n_boot=n_boot,
        seed=family_seed(feat, tgt, SEED_PAIR) + lo * 17 + hi,
    )
    return {
        "feature": feat,
        "target": tgt,
        **feature_meta(feat),
        "task_a": int(ta),
        "task_a_ja": TASK_JA[int(ta)],
        "task_b": int(tb),
        "task_b_ja": TASK_JA[int(tb)],
        **got,
    }


def run_q_supp(panel, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fam_rows = []
    cell_rows = []
    n_feat = len(panel.features)
    for fi, feat in enumerate(panel.features):
        meta = feature_meta(feat)
        x_by = panel.X[feat]
        for tgt in TARGETS:
            name = tgt["name"]
            y = panel.y[name]
            pts = observed_tasks(x_by, y)
            for tid, pt in pts.items():
                cell_rows.append(
                    {
                        "feature": feat,
                        "target": name,
                        "task": int(tid),
                        "task_ja": TASK_JA[int(tid)],
                        **meta,
                        "n": pt["n"],
                        "rho": pt["rho"],
                        "z": pt["z"],
                        "w": pt["w"],
                    }
                )
            if len(pts) < 2:
                fam_rows.append({"feature": feat, "target": name, **meta, "T": len(pts), "p": np.nan})
                continue
            got = bootstrap_q(x_by, y, pts, n_boot=n_boot, seed=family_seed(feat, name, SEED_Q))
            fam_rows.append({"feature": feat, "target": name, **meta, **got})
        if (fi + 1) % 10 == 0 or fi + 1 == n_feat:
            print(f"  Q  {fi + 1}/{n_feat} features", flush=True)
    return attach_bh_global(pd.DataFrame(fam_rows)), pd.DataFrame(cell_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/hetero")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument("--hits", type=Path, default=HIT_DEFAULT)
    ap.add_argument("--n-boot-pair", type=int, default=N_BOOT_PAIR)
    ap.add_argument("--n-boot-q", type=int, default=N_BOOT_Q)
    ap.add_argument("--n-boot-verify", type=int, default=N_BOOT_VERIFY)
    ap.add_argument("--q-supp", action="store_true", help="recompute 10-task Cochran Q (supplement)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    failed = self_check()
    if failed:
        print("self-check FAILED:", *failed, sep="\n  ", flush=True)
        raise SystemExit(1)
    print("self-check ok", flush=True)
    if args.self_check and not args.smoke and not args.q_supp:
        return

    if args.smoke:
        args.n_boot_pair = min(args.n_boot_pair, 80)
        args.n_boot_q = min(args.n_boot_q, 80)
        args.n_boot_verify = min(args.n_boot_verify, 80)
        if args.out == ROOT / "outputs/hetero":
            args.out = ROOT / "outputs/_smoke/hetero"

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cohort = load_cohort()
    dyn = load_features(args.features)
    win = pd.read_csv(args.windows)
    win["participant_id"] = win["participant_id"].astype(str)
    lab = load_labels()
    panel = build_panel(dyn, win, lab, cohort, TASKS)
    print(f"panel  n={len(cohort)}  features={len(panel.features)}", flush=True)

    if not args.hits.exists():
        raise SystemExit(f"Exp1 hits not found: {args.hits}")
    hits = pd.read_csv(args.hits)
    multi = exp1_multitask_pairs(hits)

    fig_rows = []
    contrast_rows = []
    print(f"21-pair Δz  B={args.n_boot_pair}", flush=True)
    for rec in multi.itertuples(index=False):
        picked = pick_figure_pair(rec.tasks, panel, rec.feature, rec.target)
        if picked is None:
            continue
        picked["n_hit_tasks"] = rec.n_hit_tasks
        picked["sign_flip_hits"] = rec.sign_flip
        fig_rows.append(picked)
        contrast_rows.append(
            _contrast_row(
                panel,
                rec.feature,
                rec.target,
                int(picked["task_a"]),
                int(picked["task_b"]),
                args.n_boot_pair,
            )
        )
    pairs = pd.DataFrame(fig_rows)
    contrasts = attach_bh_one(pd.DataFrame(contrast_rows), p_col="p_boot")
    merged = pairs.merge(
        contrasts[
            [
                "feature",
                "target",
                "task_a",
                "task_b",
                "delta_rho",
                "delta_z",
                "p_boot",
                "p_hw",
                "t_hw",
                "ci_rho_lo",
                "ci_rho_hi",
                "q_fdr",
                "m",
                "fdr_sig",
            ]
        ],
        on=["feature", "target", "task_a", "task_b"],
        how="left",
        suffixes=("", "_dup"),
    )
    merged = attach_pool_gains(merged)
    merged.to_csv(args.out / "pairs_21.csv", index=False)
    n_flip = int(merged["sign_flip_cap"].sum())
    n_flip_sig = int(((merged["sign_flip_cap"] == True) & (merged["fdr_sig"] == True)).sum())  # noqa: E712
    n_same_sig = int(((merged["sign_flip_cap"] == False) & (merged["fdr_sig"] == True)).sum())  # noqa: E712
    print(
        f"21 pairs  flip={n_flip} FDR-among-flip={n_flip_sig}  "
        f"same-sign FDR={n_same_sig}  pool-worse={int(merged.pool_worse.sum())}",
        flush=True,
    )
    plot_lambda_r(merged, args.out / "fig_lambda_r.png")

    flip_rows = []
    for rec in multi.itertuples(index=False):
        if not rec.sign_flip:
            continue
        y = panel.y[rec.target]
        xmap = panel.X[rec.feature]
        pos = [t for t in rec.tasks if t[2] > 0]
        neg = [t for t in rec.tasks if t[2] < 0]
        for ta, ja_a, rho_a, n_a in pos:
            for tb, ja_b, rho_b, n_b in neg:
                got = _contrast_row(panel, rec.feature, rec.target, int(ta), int(tb), args.n_boot_pair)
                in21 = (
                    ((merged["feature"] == rec.feature) & (merged["target"] == rec.target))
                    & (
                        (
                            (merged["task_a"] == int(ta)) & (merged["task_b"] == int(tb))
                        )
                        | (
                            (merged["task_a"] == int(tb)) & (merged["task_b"] == int(ta))
                        )
                    )
                )
                q21 = float(merged.loc[in21, "q_fdr"].iloc[0]) if in21.any() else float("nan")
                sig21 = bool(merged.loc[in21, "fdr_sig"].iloc[0]) if in21.any() else False
                xa, xb = xmap[int(ta)], xmap[int(tb)]
                msk = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
                n_cap = int(msk.sum())
                flip_rows.append(
                    {
                        "feature": rec.feature,
                        "target": rec.target,
                        **feature_meta(rec.feature),
                        "task_pos": int(ta),
                        "task_pos_ja": ja_a,
                        "task_neg": int(tb),
                        "task_neg_ja": ja_b,
                        "n_t_pos": float(n_a),
                        "n_t_neg": float(n_b),
                        "rho_t_pos": float(rho_a),
                        "rho_t_neg": float(rho_b),
                        "n_cap": n_cap,
                        "rho_cap_pos": got["rho_a"] if got["task_a"] == int(ta) else got["rho_b"],
                        "rho_cap_neg": got["rho_b"] if got["task_a"] == int(ta) else got["rho_a"],
                        "r_spear": got["r_spear"],
                        "delta_rho": abs(float(got["delta_rho"])),
                        "p_boot": got["p_boot"],
                        "p_hw": got["p_hw"],
                        "t_hw": got["t_hw"],
                        "ci_rho_lo": got["ci_rho_lo"],
                        "ci_rho_hi": got["ci_rho_hi"],
                        "in_pairs_21": bool(in21.any()),
                        "q_fdr_21": q21,
                        "fdr_sig_21": sig21,
                    }
                )
    flips = pd.DataFrame(flip_rows)
    flips.to_csv(args.out / "signflip_10.csv", index=False)
    print(f"sign-flip combos  {len(flips)}  (Table 1)", flush=True)

    vsum: dict = {"n_pairs": 0}
    print(f"eq3 verify  B={args.n_boot_verify}  (not a test)", flush=True)
    verify = verify_signflip_table(panel, flips, n_boot=args.n_boot_verify)
    verify.to_csv(args.out / "pool_verify_10.csv", index=False)
    plot_eq3_verify(verify, args.out / "fig_ci_forest.png")
    vsum = pool_verify_summary(verify)
    if vsum.get("n_pairs"):
        if int(vsum.get("n_match_cap") or 0) != int(vsum["n_pairs"]):
            print(
                f"WARNING  intersection ρ mismatch vs signflip rows  "
                f"{vsum.get('n_match_cap')}/{vsum['n_pairs']}",
                flush=True,
            )
        print(
            f"eq3  median |pred-obs|={vsum['median_abs_err']:.5f}  "
            f"max={vsum['max_abs_err']:.5f}  "
            f"ρz CI covers 0: {vsum['n_rho_z_covers_0']}/{vsum['n_pairs']}  "
            f"ρ+ CI excludes 0: {vsum['n_rho_pos_excludes_0']}/{vsum['n_pairs']}",
            flush=True,
        )

    if args.q_supp:
        print(f"Q supplement  B={args.n_boot_q}", flush=True)
        families, cells = run_q_supp(panel, args.n_boot_q)
        families.to_csv(args.out / "families.csv", index=False)
        cells.to_csv(args.out / "task_rho.csv", index=False)
        n_sig = int(families["fdr_sig"].sum()) if "fdr_sig" in families else 0
        print(
            f"Q  FDR={n_sig}  uncorrected p<0.05={int((families.p < 0.05).sum())}  "
            f"q min={float(families.q_fdr.min()):.3f}",
            flush=True,
        )
    elif (args.out / "families.csv").exists():
        fam = pd.read_csv(args.out / "families.csv")
        print(
            f"Q supplement kept  {args.out / 'families.csv'}  "
            f"FDR={int(fam.fdr_sig.sum()) if 'fdr_sig' in fam else '?'}  "
            f"q min={float(fam.q_fdr.min()) if 'q_fdr' in fam else float('nan'):.3f}",
            flush=True,
        )

    print("findings (λ-r, duration, Q ranks, initiator)", flush=True)
    from ados_ffm.exp3_findings import run as run_findings

    findings = run_findings(
        args.out,
        features=args.features,
        windows=args.windows,
        hits=args.hits,
        n_boot_maxstat=min(args.n_boot_pair, 80) if args.smoke else args.n_boot_pair,
        skip_initiators=bool(args.smoke),
        skip_maxstat=True,
    )

    meta = {
        "role": "exp3_pairs_on_exp1_multitask_hits",
        "n_cohort": int(len(cohort)),
        "n_pairs_21": int(len(merged)),
        "n_sign_flip_21": n_flip,
        "n_sign_flip_fdr": n_flip_sig,
        "n_same_sign_fdr": n_same_sig,
        "n_pool_worse": int(merged.pool_worse.sum()),
        "n_signflip_combos": int(len(flips)),
        "bh_family": "the_21_exp1_multitask_pairs",
        "q_level": 0.05,
        "n_boot_pair": int(args.n_boot_pair),
        "pair_null": "H0_rhoA_eq_rhoB_subject_bootstrap_BCa",
        "pair_analytic": "hotelling_williams_on_spearman",
        "conditioned_on": "exp1_fdr_hits_appearing_in_ge2_tasks",
        "headline": "where_pairs_fell_in_lambda_r_not_the_21_tests",
        "q_supplement": "families.csv (10-task Cochran Q; report p ranks, not FDR 0 alone)",
        "pool_verify": {
            "role": "eq3_prediction_vs_measured_on_signflip_10",
            "not_a_test": True,
            "definition": "P1_rank_then_pool_within_intersection",
            "ci": "subject_bootstrap_percentile_rerank_inside",
            "n_boot": int(args.n_boot_verify),
            **vsum,
        },
        "seconds": float(time.perf_counter() - t0),
        "self_check_failed": failed,
        "findings": findings,
    }
    (args.out / "meta.json").write_text(
        json.dumps(_jsonable(meta), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {args.out}  ({meta['seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
