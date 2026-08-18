#!/usr/bin/env python
"""Leave-one-out group residualization, then redo Exp1–4 tests (§8).

    ./venv/bin/python -u scripts/run_group_check.py
    ./venv/bin/python -u scripts/run_group_check.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_ffm.data import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_WINDOWS,
    TASK_JA,
    TASKS,
    TARGETS,
    group_of,
    load_cohort,
    load_features,
    load_labels,
)
from ados_ffm.exp1_univariate import (  # noqa: E402
    N_PERM,
    SEED_PERM,
    attach_fdr,
    catalog_cols,
    feature_payloads,
    merge_task_catalog,
    perm_one,
    stable_int,
    y_map_for_task,
)
from ados_ffm.exp2_ridge import (  # noqa: E402
    attach_fdr_task_target,
    prepare_jobs,
    run_one,
)
from ados_ffm.ridge_cv import SOURCES  # noqa: E402
from ados_ffm.exp3_transplant import run_side as run_exp3_side  # noqa: E402
from ados_ffm.exp3_transplant import tests as tests_exp3  # noqa: E402
from ados_ffm.exp4_pool import run_side as run_exp4_side  # noqa: E402
from ados_ffm.exp4_pool import tests as tests_exp4  # noqa: E402
from ados_ffm.exp34_labels import N_LABEL_PERM  # noqa: E402
from ados_ffm.group_residual import (  # noqa: E402
    classify,
    group_counts,
    loo_center,
    residualize_frame,
    small_group,
)

TGT = {t["name"]: t for t in TARGETS}


def _tgt_index(name: str) -> int:
    return {"SA": 0, "RRB": 1, "CSS": 2}[name]


def pack_task(dyn, new, lab, cohort, task_id: int, target: dict):
    frame = merge_task_catalog(dyn, new, task_id, cohort)
    if frame.empty:
        return None
    ymap = y_map_for_task(frame, lab, target)
    cols = catalog_cols(list(frame.columns))
    payloads = feature_payloads(frame, ymap, cols)
    if not payloads:
        return None
    return payloads


def residualize_payload(pl: dict, lab: pd.DataFrame) -> dict | None:
    g = np.array([group_of(p, lab) for p in pl["pids"]])
    x = loo_center(pl["x"], g)
    y = loo_center(pl["y"], g)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 8:
        return None
    out = dict(pl)
    out["x"] = x[ok]
    out["y"] = y[ok]
    out["pids"] = [p for p, k in zip(pl["pids"], ok) if k]
    out["n"] = int(ok.sum())
    out["groups"] = g[ok]
    return out


def exp1_check(dyn, new, lab, cohort, n_perm: int, n_jobs: int) -> pd.DataFrame:
    cand = pd.read_csv(ROOT / "outputs/exp1/candidates.csv")
    hits = pd.read_csv(ROOT / "outputs/exp1/hits_main.csv")
    packed: dict[tuple[int, str], list] = {}
    for task in TASKS:
        for t in TARGETS:
            packed[(int(task), t["name"])] = pack_task(dyn, new, lab, cohort, int(task), t)

    def one(row: dict) -> dict | None:
        key = (int(row["task"]), row["target"])
        payloads = packed[key]
        if not payloads:
            return None
        pl = next((p for p in payloads if p["feature"] == row["feature"]), None)
        if pl is None:
            return None
        pl2 = residualize_payload(pl, lab)
        if pl2 is None:
            return None
        seed = SEED_PERM + int(row["task"]) * 300 + _tgt_index(row["target"]) * 40
        seed += stable_int(str(row["feature"]))
        fit = perm_one(pl2["x"], pl2["y"], n_perm=n_perm, seed=seed)
        gc = group_counts(pl2["groups"])
        return {
            "task": int(row["task"]),
            "task_ja": row["task_ja"],
            "source": row["source"],
            "target": row["target"],
            "feature": row["feature"],
            "n": pl2["n"],
            **gc,
            "small_group": small_group(gc["n_preterm"], gc["n_other"]),
            "rho": fit["rho"],
            "p": fit["p"],
        }

    jobs = cand.to_dict("records")
    got = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(r) for r in jobs)
    redo = pd.DataFrame([r for r in got if r is not None])
    redo = attach_fdr(redo)
    hit_key = ["task", "source", "target", "feature"]
    old = hits[hit_key + ["rho", "q_fdr", "fdr_sig"]].rename(
        columns={"rho": "rho_old", "q_fdr": "q_old", "fdr_sig": "sig_old"}
    )
    m = old.merge(
        redo,
        on=hit_key,
        how="left",
        suffixes=("", ""),
    )
    # redo columns already named rho, q_fdr, fdr_sig
    rows = []
    for r in m.itertuples(index=False):
        d = r._asdict() if hasattr(r, "_asdict") else dict(zip(m.columns, r))
        cls = classify(d["rho_old"], d.get("rho", float("nan")), bool(d["sig_old"]), bool(d.get("fdr_sig", False)))
        rows.append({**d, **cls, "rho_new": d.get("rho"), "q_new": d.get("q_fdr"), "sig_new": d.get("fdr_sig")})
    return pd.DataFrame(rows)


def exp2_check(dyn, new, lab, cohort, n_perm: int, n_jobs: int) -> pd.DataFrame:
    hits = pd.read_csv(ROOT / "outputs/exp1/hits_main.csv")
    old = pd.read_csv(ROOT / "outputs/exp2/cells.csv")
    jobs, _ = prepare_jobs(hits, dyn, new, lab, cohort, TARGETS, TASKS, SOURCES)
    for j in jobs:
        j["frame"] = residualize_frame(j["frame"], [*j["cols"], "y"])
        j["frame"] = j["frame"].dropna(subset=[*j["cols"], "y"]).reset_index(drop=True)
        j["n"] = int(len(j["frame"]))
        gc = group_counts(j["frame"]["group"])
        j["n_preterm"] = gc["n_preterm"]
        j["n_other"] = gc["n_other"]
        j["small_group"] = small_group(gc["n_preterm"], gc["n_other"])
    jobs = [j for j in jobs if j["n"] >= 12]
    got = Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_one)(j, n_perm) for j in jobs)
    redo = pd.DataFrame(got)
    extra = pd.DataFrame(
        [
            {
                "task": j["task"],
                "source": j["source"],
                "target": j["target"],
                "n_preterm": j["n_preterm"],
                "n_other": j["n_other"],
                "small_group": j["small_group"],
            }
            for j in jobs
        ]
    )
    redo = redo.merge(extra, on=["task", "source", "target"], how="left")
    redo = attach_fdr_task_target(redo)
    old_s = old[["task", "source", "target", "rho", "q_fdr", "fdr_sig"]].rename(
        columns={"rho": "rho_old", "q_fdr": "q_old", "fdr_sig": "sig_old"}
    )
    m = old_s.merge(
        redo.rename(columns={"rho": "rho_new", "q_fdr": "q_new", "fdr_sig": "sig_new"}),
        on=["task", "source", "target"],
        how="left",
    )
    cls = [
        classify(r.rho_old, r.rho_new, bool(r.sig_old), bool(r.sig_new) if pd.notna(r.sig_new) else False)
        for r in m.itertuples(index=False)
    ]
    return pd.concat([m, pd.DataFrame(cls)], axis=1)


def _residual_exp3_side(row, tabs, lab, target):
    from ados_ffm.ridge_cv import cv_schedule, run_ridge_cols_cv
    from ados_ffm.exp34_labels import directed_seed, people_cols

    onto = int(row["task_onto"])
    own, other = str(row["col_own"]), str(row["col_from"])
    tab = tabs.get(onto)
    if tab is None or tab.empty:
        return None
    fr = people_cols(tab, lab, [own, other], target)
    if fr is None:
        return None
    fr = residualize_frame(fr, [own, other, "y"]).dropna(subset=[own, other, "y"])
    if len(fr) < 12:
        return None
    n = int(len(fr))
    sched = cv_schedule(n)
    if sched is None:
        return None
    n_splits, n_repeats = sched
    seed = directed_seed(onto, int(row["task_from"]), str(row["source"]), str(row["target"]))
    rho_own = run_ridge_cols_cv(fr, [own], n_splits, n_repeats, seed)["rho"]
    rho_from = run_ridge_cols_cv(fr, [other], n_splits, n_repeats, seed)["rho"]
    gc = group_counts(fr["group"])
    return {
        **{k: row[k] for k in row},
        "n": n,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "rho_own": float(rho_own),
        "rho_from": float(rho_from),
        "delta3": float(rho_own) - float(rho_from),
        **gc,
        "small_group": small_group(gc["n_preterm"], gc["n_other"]),
    }


def _residual_exp4_side(row, tabs, lab, target):
    from ados_ffm.ridge_cv import cv_schedule, run_ridge_cols_cv
    from ados_ffm.exp34_labels import directed_seed, people_frame, people_ready

    onto, other = int(row["task_onto"]), int(row["task_from"])
    col = str(row["col_own"])
    ta, tb = tabs.get(onto), tabs.get(other)
    if ta is None or tb is None or ta.empty or tb.empty:
        return None
    pa = people_frame(ta, lab, [col], target)
    pb = people_frame(tb, lab, [col], target)
    if pa is None or pb is None:
        return None
    both = sorted(set(pa["participant_id"]) & set(pb["participant_id"]))
    pa = pa[pa["participant_id"].isin(both)].sort_values("participant_id").reset_index(drop=True)
    pb = pb[pb["participant_id"].isin(both)].sort_values("participant_id").reset_index(drop=True)
    pa = residualize_frame(pa, [col, "y"])
    pb = residualize_frame(pb, [col, "y"])
    ok = np.isfinite(pa[col]) & np.isfinite(pb[col]) & np.isfinite(pa["y"])
    pa, pb = pa.loc[ok].reset_index(drop=True), pb.loc[ok].reset_index(drop=True)
    fr = people_ready(pa)
    if fr is None:
        return None
    mix = fr.copy()
    mix[col] = 0.5 * (pa[col].to_numpy(dtype=float) + pb[col].to_numpy(dtype=float))
    n = int(len(fr))
    n_splits, n_repeats = cv_schedule(n)
    seed = directed_seed(onto, other, str(row["source"]), str(row["target"]))
    rho_solo = run_ridge_cols_cv(fr, [col], n_splits, n_repeats, seed)["rho"]
    rho_mix = run_ridge_cols_cv(mix, [col], n_splits, n_repeats, seed)["rho"]
    gc = group_counts(fr["group"])
    return {
        **{k: row[k] for k in row},
        "n": n,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "rho_solo": float(rho_solo),
        "rho_mix": float(rho_mix),
        "delta4": float(rho_mix) - float(rho_solo),
        **gc,
        "small_group": small_group(gc["n_preterm"], gc["n_other"]),
    }


def exp34_check(which: str, dyn, new, lab, cohort, n_perm: int, n_jobs: int):
    old_main = pd.read_csv(ROOT / f"outputs/{which}/main_test.csv")
    old_sides = pd.read_csv(ROOT / f"outputs/{which}/sides.csv")
    labels = pd.read_csv(ROOT / f"outputs/{which}/near_far_labels.csv")
    from ados_ffm.exp34_labels import expand_sides

    sides = expand_sides(labels)
    task_ids = sorted(set(sides["task_onto"].astype(int)) | set(sides["task_from"].astype(int)))
    tabs = {tid: merge_task_catalog(dyn, new, tid, cohort) for tid in task_ids}
    fn = _residual_exp3_side if which == "exp3" else _residual_exp4_side
    jobs = sides.to_dict("records")
    got = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(fn)(row, tabs, lab, TGT[row["target"]]) for row in jobs
    )
    redo_sides = pd.DataFrame([r for r in got if r is not None])
    delta = "delta3" if which == "exp3" else "delta4"
    testers = tests_exp3 if which == "exp3" else tests_exp4
    main, _ = testers(redo_sides, n_perm=n_perm) if not redo_sides.empty else (pd.DataFrame(), pd.DataFrame())
    old_s = old_main.rename(
        columns={
            "mean_delta_far": "stat_old",
            "q_fdr": "q_old",
            "fdr_sig": "sig_old",
            "p": "p_old",
        }
    )
    new_s = main.rename(
        columns={
            "mean_delta_far": "stat_new",
            "q_fdr": "q_new",
            "fdr_sig": "sig_new",
            "p": "p_new",
        }
    )
    keep_old = ["source", "target", "n_far", "n_near", "stat_old", "q_old", "sig_old", "p_old"]
    keep_new = ["source", "target", "stat_new", "q_new", "sig_new", "p_new", "n_far", "n_near"]
    m = old_s[keep_old].merge(
        new_s[keep_new], on=["source", "target"], how="left", suffixes=("_oldn", "_newn")
    )
    cls = [
        classify(
            r.stat_old,
            r.stat_new if pd.notna(r.stat_new) else float("nan"),
            bool(r.sig_old),
            bool(r.sig_new) if pd.notna(r.sig_new) else False,
        )
        for r in m.itertuples(index=False)
    ]
    out = pd.concat([m, pd.DataFrame(cls)], axis=1)
    return out, redo_sides


def summarize(df: pd.DataFrame, reported_mask: pd.Series | None = None) -> dict:
    sub = df if reported_mask is None else df.loc[reported_mask]
    return {
        "n_reported": int(len(sub)),
        "n_collapsed": int(sub["collapsed"].sum()) if "collapsed" in sub else 0,
        "n_sign_flip": int(sub["sign_flip"].sum()) if "sign_flip" in sub else 0,
        "n_lost_fdr": int(sub["lost_fdr"].sum()) if "lost_fdr" in sub else 0,
        "n_weakened": int(sub["weakened"].sum()) if "weakened" in sub else 0,
        "n_ok": int((sub["verdict"] == "崩れず").sum()) if "verdict" in sub else 0,
        "n_small_group": int(sub["small_group"].sum()) if "small_group" in sub else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/group_check")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-perm-34", type=int, default=N_LABEL_PERM)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-exp2", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if args.smoke:
        args.n_perm = min(args.n_perm, 5)
        args.n_perm_34 = min(args.n_perm_34, 5)
        if args.out == ROOT / "outputs/group_check":
            args.out = ROOT / "outputs/_smoke/group_check"
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    lab = load_labels()
    dyn = load_features()
    dyn = dyn[dyn["participant_id"].astype(str).isin(cohort)].copy()
    new = pd.read_csv(DEFAULT_WINDOWS)
    new["participant_id"] = new["participant_id"].astype(str)
    t0 = time.time()

    print("exp1 residual …", flush=True)
    e1 = exp1_check(dyn, new, lab, cohort, args.n_perm, args.n_jobs)
    e1.to_csv(args.out / "exp1_hits.csv", index=False)

    if args.skip_exp2:
        e2 = pd.DataFrame()
    else:
        print("exp2 residual …", flush=True)
        e2 = exp2_check(dyn, new, lab, cohort, args.n_perm, args.n_jobs)
        e2.to_csv(args.out / "exp2_cells.csv", index=False)

    print("exp3 residual …", flush=True)
    e3, s3 = exp34_check("exp3", dyn, new, lab, cohort, args.n_perm_34, args.n_jobs)
    e3.to_csv(args.out / "exp3_main.csv", index=False)
    s3.to_csv(args.out / "exp3_sides.csv", index=False)

    print("exp4 residual …", flush=True)
    e4, s4 = exp34_check("exp4", dyn, new, lab, cohort, args.n_perm_34, args.n_jobs)
    e4.to_csv(args.out / "exp4_main.csv", index=False)
    s4.to_csv(args.out / "exp4_sides.csv", index=False)

    e2_rep = e2[e2["sig_old"]] if not e2.empty and "sig_old" in e2.columns else e2
    summary = {
        "n_perm": args.n_perm,
        "n_perm_34": args.n_perm_34,
        "smoke": bool(args.smoke),
        "seconds": round(time.time() - t0, 1),
        "exp1_110": summarize(e1),
        "exp2_40": summarize(e2_rep) if not e2.empty else {"skipped": True},
        "exp3_9": summarize(e3),
        "exp4_9": summarize(e4),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"done  {summary['seconds']:.1f}s  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
