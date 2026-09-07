#!/usr/bin/env python
"""Old-vs-new report for the task-interval merge-rule change (2026-09-07).

Prints the eleven quantities the paper cites, side by side: the archived
convex-hull run in ``outputs/old/20260907_hull_merge/`` against the current
union-of-stretches run in ``outputs/``. Anything that could not be computed is
reported as missing rather than filled in.

    ./venv/bin/python scripts/compare_hull_vs_union.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ados_extract.task_segments import (  # noqa: E402
    load_task_segments_json,
    merge_spans,
    parse_time_text,
)
from ados_ffm.data import TASK_JA, load_cohort  # noqa: E402

OLD = ROOT / "outputs/old/20260907_hull_merge"
NEW = ROOT / "outputs"
MISSING = "（出なかった）"


def _read_csv(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return MISSING
    if isinstance(v, float) and not np.isfinite(v):
        return MISSING
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _line(label: str, old, new, nd: int = 4) -> None:
    o, n = _fmt(old, nd), _fmt(new, nd)
    tag = "  変化なし" if o == n else ""
    print(f"  {label:<44} 旧 {o:>12}   新 {n:>12}{tag}")


def head(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


# --------------------------------------------------------------------------
def item1_rows() -> None:
    head(1, "参加者×課題の行数（20秒未満の除外後）と課題ごとの人数")
    o = _read_csv(OLD / "features/dynamics/features_task.csv", dtype={"participant_id": str})
    n = _read_csv(NEW / "features/dynamics/features_task.csv", dtype={"participant_id": str})
    if o is None or n is None:
        print(f"  {MISSING}: 特徴CSVが無い")
        return
    _line("総行数", int(len(o)), int(len(n)), 0)
    _line("参加者数", int(o.participant_id.nunique()), int(n.participant_id.nunique()), 0)
    print(f"\n  {'課題':<22}{'旧 n':>7}{'新 n':>7}")
    for tid in sorted(set(o.task_id) | set(n.task_id)):
        co = int((o.task_id == tid).sum())
        cn = int((n.task_id == tid).sum())
        flag = "" if co == cn else "   <-- 変化"
        print(f"  {tid:>2} {TASK_JA.get(tid, ''):<19}{co:>7}{cn:>7}{flag}")


def item2_durations() -> None:
    head(2, "課題区間の長さ（Supplementary Table S1 対応）")
    o = _read_csv(OLD / "hetero/duration_by_task.csv")
    n = _read_csv(NEW / "hetero/duration_by_task.csv")
    if o is None or n is None:
        print(f"  {MISSING}: duration_by_task.csv が無い")
        return
    o = o.set_index("task")
    n = n.set_index("task")
    print(f"  {'課題':<22}{'旧 n':>5}{'旧中央':>8}{'旧最小':>8}{'旧最大':>9}"
          f"{'新 n':>6}{'新中央':>8}{'新最小':>8}{'新最大':>9}")
    for tid in sorted(set(o.index) | set(n.index)):
        ro = o.loc[tid] if tid in o.index else None
        rn = n.loc[tid] if tid in n.index else None
        def g(r, c):
            return f"{r[c]:.0f}" if r is not None else "-"
        changed = ro is None or rn is None or any(
            abs(float(ro[c]) - float(rn[c])) > 0.5
            for c in ("n", "median_sec", "min_sec", "max_sec")
        )
        print(f"  {tid:>2} {TASK_JA.get(tid, ''):<19}"
              f"{g(ro,'n'):>5}{g(ro,'median_sec'):>8}{g(ro,'min_sec'):>8}{g(ro,'max_sec'):>9}"
              f"{g(rn,'n'):>6}{g(rn,'median_sec'):>8}{g(rn,'min_sec'):>8}{g(rn,'max_sec'):>9}"
              + ("   <-- 変化" if changed else ""))


def item3_r_distribution() -> None:
    head(3, "課題間相関 r の分布")
    o = _read_json(OLD / "hetero/findings_meta.json")
    n = _read_json(NEW / "hetero/findings_meta.json")
    if o is None or n is None:
        print(f"  {MISSING}: findings_meta.json が無い")
        return
    for label, key, nd in (
        ("通り数 (n_pairs_valid)", "n_pairs_valid", 0),
        ("特徴数 (n_features)", "n_features", 0),
        ("中央値", "r_median", 4),
        ("第1四分位", "r_q25", 4),
        ("第3四分位", "r_q75", 4),
        ("r < 0.2 の割合", "frac_r_lt_0.2", 4),
        ("r < 0 の割合", "frac_r_lt_0", 4),
    ):
        _line(label, o.get(key), n.get(key), nd)


def item4_exp1() -> None:
    head(4, "実験1のスクリーニング")
    o = _read_json(OLD / "exp1/meta.json")
    n = _read_json(NEW / "exp1/meta.json")
    if o is None or n is None:
        print(f"  {MISSING}: exp1/meta.json が無い")
        return
    for label, key in (
        ("検定数 (n_stage_a)", "n_stage_a"),
        ("安定性通過 (n_candidates)", "n_candidates"),
        ("BH補正通過 (n_main_fdr_hits)", "n_main_fdr_hits"),
    ):
        _line(label, o.get(key), n.get(key), 0)


TABLE3_COLS = [
    "feature", "target", "task_a", "task_a_ja", "task_b", "task_b_ja",
    "rho_a_orient", "rho_b_orient", "ci_rho_lo", "ci_rho_hi", "r_spear", "n_cap",
]


def item5_pairs() -> None:
    head(5, "2活動以上で生き残った組合せと符号反転（Table 3）")
    o = _read_csv(OLD / "hetero/pairs_21.csv")
    n = _read_csv(NEW / "hetero/pairs_21.csv")
    if o is None or n is None:
        print(f"  {MISSING}: pairs_21.csv が無い")
        return
    _line("生存した組合せ", int(len(o)), int(len(n)), 0)
    _line("うち符号反転", int(o.sign_flip_cap.sum()), int(n.sign_flip_cap.sum()), 0)
    for tag, df in (("旧", o), ("新", n)):
        f = df[df.sign_flip_cap.astype(bool)]
        print(f"\n  --- {tag}：符号反転の一覧（{len(f)}件） ---")
        if f.empty:
            print(f"    {MISSING}")
            continue
        print(f.reindex(columns=TABLE3_COLS).to_string(index=False))
    ko = {(r.feature, r.target) for r in o.itertuples() if r.sign_flip_cap}
    kn = {(r.feature, r.target) for r in n.itertuples() if r.sign_flip_cap}
    print(f"\n  旧のみ: {sorted(ko - kn) or 'なし'}")
    print(f"  新のみ: {sorted(kn - ko) or 'なし'}")


def item6_lambda() -> None:
    head(6, "λ = 弱いほう / 強いほう の範囲（符号反転組）")
    for tag, path in (("旧", OLD), ("新", NEW)):
        df = _read_csv(path / "hetero/pairs_21.csv")
        if df is None:
            print(f"  {tag}: {MISSING}")
            continue
        lam = df.loc[df.sign_flip_cap.astype(bool), "lambda"].to_numpy(dtype=float)
        lam = lam[np.isfinite(lam)]
        if lam.size == 0:
            print(f"  {tag}: {MISSING}（符号反転が0件）")
            continue
        print(f"  {tag}: n={lam.size}  範囲 {lam.max():.4f} 〜 {lam.min():.4f}  "
              f"−0.88 以下 {int((lam <= -0.88).sum())}件")
        print(f"      値: {', '.join(f'{v:.3f}' for v in sorted(lam, reverse=True))}")


def item7_rank_avg() -> None:
    head(7, "活動内順位を平均したときの相関の区間（ゼロを含む数）")
    o = _read_json(OLD / "hetero/findings_meta.json")
    n = _read_json(NEW / "hetero/findings_meta.json")
    if o is None or n is None:
        print(f"  {MISSING}: findings_meta.json が無い")
        return
    _line("反転組合せ数 (n_pairs)", o.get("n_pairs"), n.get("n_pairs"), 0)
    _line("rho_z の区間がゼロを含む数", o.get("n_rho_z_covers_0"), n.get("n_rho_z_covers_0"), 0)
    _line("rho_plus の区間がゼロを除く数", o.get("n_rho_plus_excludes_0"),
          n.get("n_rho_plus_excludes_0"), 0)


def item8_homogeneity() -> None:
    head(8, "無選択の等質性検定")
    o = _read_json(OLD / "hetero/findings_meta.json")
    n = _read_json(NEW / "hetero/findings_meta.json")
    if o is None or n is None:
        print(f"  {MISSING}: findings_meta.json が無い")
        return
    _line("組合せ数 (n_families)", o.get("n_families"), n.get("n_families"), 0)
    stats = {}
    for tag, path in (("旧", OLD), ("新", NEW)):
        df = _read_csv(path / "hetero/families.csv")
        if df is None:
            print(f"  {tag}: {MISSING}（families.csv が無い）")
            stats[tag] = (None, None, None, None)
            continue
        stats[tag] = (
            int(len(df)),
            int(df["fdr_sig"].astype(bool).sum()) if "fdr_sig" in df else None,
            float(df["q_fdr"].min()) if "q_fdr" in df else None,
            float(np.nanmedian(df["I2"])) if "I2" in df else None,
        )
    for i, (label, nd) in enumerate(
        (("families.csv の行数", 0), ("BH補正後に生存", 0), ("最小 q", 4), ("中央値 I²", 4))
    ):
        _line(label, stats["旧"][i], stats["新"][i], nd)


def item9_exp2() -> None:
    head(9, "実験2")
    o = _read_json(OLD / "exp2/run_meta.json")
    n = _read_json(NEW / "exp2/run_meta.json")
    if o is None or n is None:
        print(f"  {MISSING}: exp2/run_meta.json が無い（実験2が未実行）")
        return
    _line("補正族の数 (n_jobs_ran)", o.get("n_jobs_ran"), n.get("n_jobs_ran"), 0)
    _line("棄却された族 (n_fdr)", o.get("n_fdr"), n.get("n_fdr"), 0)
    for tag, path in (("旧", OLD), ("新", NEW)):
        c = _read_csv(path / "exp2/cells.csv")
        if c is not None and "q_family" in c:
            nf = int(c["q_family"].nunique())
            print(f"  {tag} 補正族数 {nf}，完全帰無での期待棄却数 {0.05 * nf:.1f}")
    for tag, path in (("旧", OLD), ("新", NEW)):
        cells = _read_csv(path / "exp2/cells.csv")
        if cells is None:
            print(f"  {tag}: {MISSING}（cells.csv が無い）")
            continue
        # top1_frac is the share of folds whose rank-1 feature was the same one;
        # 1.0 means every fold picked the same feature.
        stable = int((cells["top1_frac"] >= 1.0).sum()) if "top1_frac" in cells else None
        print(f"  {tag}: 全fold同一特徴 {_fmt(stable, 0)}/{len(cells)} セル")
        hits = cells[cells["fdr_sig"].astype(bool)] if "fdr_sig" in cells else None
        if hits is not None and not hits.empty:
            cols = [c for c in ("task", "task_ja", "source", "target", "cols", "rho", "q_fdr")
                    if c in hits.columns]
            print(hits[cols].to_string(index=False))


def item10_duration_y() -> None:
    head(10, "区間長とアウトカムの相関")
    o = _read_csv(OLD / "hetero/duration_y.csv")
    n = _read_csv(NEW / "hetero/duration_y.csv")
    if o is None or n is None:
        print(f"  {MISSING}: duration_y.csv が無い")
        return
    o = o.set_index(["task", "target"])["rho"]
    n = n.set_index(["task", "target"])["rho"]
    print(f"  {'課題':<22}{'target':>7}{'旧 rho':>10}{'新 rho':>10}")
    for k in sorted(set(o.index) | set(n.index)):
        vo = float(o.get(k, np.nan))
        vn = float(n.get(k, np.nan))
        mark = "" if np.isclose(vo, vn, atol=5e-4) else "   <-- 変化"
        star = " *" if k[0] in (6, 8) and k[1] in ("SA", "CSS") else ""
        print(f"  {k[0]:>2} {TASK_JA.get(k[0], ''):<19}{k[1]:>7}{vo:>10.4f}{vn:>10.4f}{mark}{star}")
    print("  * 本文が引用している本(6)・誕生日(8) の SA / CSS")


def item11_overlap() -> None:
    head(11, "重なりの再計数")
    ids = load_cohort()
    raw = load_task_segments_json()

    def stretches(pid: str) -> dict[int, list[tuple[float, float]]]:
        by: dict[int, list[tuple[float, float]]] = {}
        for r in raw.get(pid, []):
            t0s, t1s = str(r.get("t0", "")).strip(), str(r.get("t1", "")).strip()
            if not t0s or not t1s:
                continue
            t0, t1 = parse_time_text(t0s), parse_time_text(t1s)
            if t1 > t0:
                by.setdefault(int(r["task_id"]), []).append((t0, t1))
        return by

    def overlap(a, b) -> float:
        return sum(max(0.0, min(b1, b2) - max(a1, a2)) for a1, b1 in a for a2, b2 in b)

    for label, mode in (("旧（凸包）", "hull"), ("新（和集合）", "union")):
        people, pairs = set(), []
        for pid in ids:
            by = stretches(pid)
            cur = {
                tid: ([(min(a for a, _ in ivs), max(b for _, b in ivs))]
                      if mode == "hull" else merge_spans(ivs))
                for tid, ivs in by.items()
            }
            for t1, t2 in combinations(sorted(cur), 2):
                sec = overlap(cur[t1], cur[t2])
                if sec > 0:
                    people.add(pid)
                    pairs.append((pid, t1, t2, sec))
        print(f"\n  {label}: {len(people)}名 / {len(pairs)}対")
        for pid, t1, t2, sec in sorted(pairs, key=lambda x: -x[3])[:8]:
            print(f"    {pid} task{t1}({TASK_JA[t1]}) x task{t2}({TASK_JA[t2]}): {sec:.0f}s")
        if len(pairs) > 8:
            print(f"    ... 他 {len(pairs) - 8} 対")


def main() -> None:
    print("課題区間の併合規則：凸包 → 和集合　新旧比較")
    print(f"旧 = {OLD}\n新 = {NEW}")
    for fn in (item1_rows, item2_durations, item3_r_distribution, item4_exp1,
               item5_pairs, item6_lambda, item7_rank_avg, item8_homogeneity,
               item9_exp2, item10_duration_y, item11_overlap):
        fn()


if __name__ == "__main__":
    main()
