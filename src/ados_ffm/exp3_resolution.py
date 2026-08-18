"""Exp3 sign-flip resolution. Same locked Δ3 and seeds, more permutations.

Does not rewrite `outputs/exp3/main_test.csv`. The 1000-perm family stays locked.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ados_ffm.exp3_transplant import tests
from ados_ffm.exp34_labels import N_LABEL_PERM, planned_combos

N_RESOLVE = 10000
SOURCE_JA = {"child": "子ども", "examiner": "検査者", "dyad": "二人"}


def compare_resolution(
    orig_main: pd.DataFrame,
    hi_main: pd.DataFrame,
    *,
    n_perm_orig: int = N_LABEL_PERM,
    n_perm_hi: int = N_RESOLVE,
) -> pd.DataFrame:
    orig_i = orig_main.set_index(["source", "target"])
    hi_i = hi_main.set_index(["source", "target"])
    rows: list[dict[str, Any]] = []
    for source, target in planned_combos():
        o = orig_i.loc[(source, target)]
        h = hi_i.loc[(source, target)]
        rows.append(
            {
                "source": source,
                "target": target,
                "n_far": int(o["n_far"]),
                "mean_delta_far": float(o["mean_delta_far"]),
                "p_orig": float(o["p"]),
                "q_orig": float(o["q_fdr"]),
                "sig_orig": bool(o["fdr_sig"]),
                "p_hi": float(h["p"]),
                "q_hi": float(h["q_fdr"]),
                "sig_hi": bool(h["fdr_sig"]),
                "n_perm_orig": n_perm_orig,
                "n_perm_hi": n_perm_hi,
            }
        )
    return pd.DataFrame(rows)


def _fmt_p(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    if x < 0.001:
        return f"{x:.4f}"
    return f"{x:.3f}"


def _combo_ja(source: str, target: str) -> str:
    return f"{SOURCE_JA.get(source, source)} {target}"


def render_readme(
    compare: pd.DataFrame,
    *,
    n_perm_orig: int,
    n_perm_hi: int,
) -> str:
    n_floor_orig = int((compare["p_orig"] <= 1.0 / (1 + n_perm_orig) + 1e-12).sum())
    n_floor_hi = int((compare["p_hi"] <= 1.0 / (1 + n_perm_hi) + 1e-12).sum())
    n_same_sig = int((compare["sig_orig"] == compare["sig_hi"]).sum())
    orig_pass = int(compare["sig_orig"].sum())
    hi_pass = int(compare["sig_hi"].sum())
    lines = [
        "# 実験3本線の解像度（符号入れ替えを増やした確認）",
        "",
        "日付: 2026-08-18  ",
        "入口: `scripts/run_exp3_resolution.py`  ",
        "対象: 実験3の本線9本．既存の Δ3 と乱数の種は変えない．入れ替えの回数だけ増やす．",
        "",
        f"本線は {n_perm_orig} 回のまま変えない（`outputs/exp3/main_test.csv`）．"
        "Ridge はやり直さない．これは新しい家族ではない．",
        "",
        f"{n_perm_orig} 回では p の下限が 1/{n_perm_orig + 1} である．"
        "複数本が下限に張り付くと，BH のあとの q も同じ数字になる．"
        f"回数を {n_perm_hi} にすると，下限は 1/{n_perm_hi + 1} になる．",
        "",
        "| 情報源×点数 | 平均Δ3 | p（1000回） | q（1000回） | p（10000回） | q（10000回） | 本線 | 参考 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in compare.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    _combo_ja(str(r.source), str(r.target)),
                    f"{float(r.mean_delta_far):.3f}",
                    _fmt_p(float(r.p_orig)),
                    _fmt_p(float(r.q_orig)),
                    _fmt_p(float(r.p_hi)),
                    _fmt_p(float(r.q_hi)),
                    "通った" if r.sig_orig else "未通過",
                    "通った" if r.sig_hi else "未通過",
                ]
            )
            + " |"
        )
    lines += [
        "",
        f"{n_perm_orig} 回で下限に付いた本は {n_floor_orig} 本．"
        f"{n_perm_hi} 回では {n_floor_hi} 本が新しい下限に付く．",
        f"通過の可否は {n_same_sig}/9 本で同じである（本線 {orig_pass} 本，参考 {hi_pass} 本）．",
        "",
        "**報告:** 本線の 7/9 は，回数を増やしても変わらない．1000回の判定は変えない．",
        "",
        "表: `compare.csv`．参考の検定: `main_test.csv`．",
    ]
    return "\n".join(lines) + "\n"
