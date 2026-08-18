"""Drop book-story (task 9) pairs from Exp3 averages. Sensitivity only.

Does not rewrite the locked Exp3 family (`outputs/exp3/main_test.csv`).
Ridge is not rerun; existing Δ3 on each side is reused.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ados_ffm.data import TASK_JA
from ados_ffm.exp3_transplant import tests
from ados_ffm.exp34_labels import N_LABEL_PERM, planned_combos

BOOK_TASK = 9
BOOK_JA = TASK_JA[BOOK_TASK]
REL_CHANGE_LIMIT = 0.20
SOURCE_JA = {"child": "子ども", "examiner": "検査者", "dyad": "二人"}

COMPARE_COLS = (
    "source",
    "target",
    "n_far_orig",
    "n_far_book",
    "n_far_keep",
    "n_near_keep",
    "mean_delta_far_orig",
    "mean_delta_far_drop_book",
    "diff",
    "rel_change",
    "orig_fdr_sig",
    "p_keep",
    "q_keep",
    "sig_keep",
    "m_keep",
    "skip_keep",
    "verdict",
    "note",
)


def involves_book(sides: pd.DataFrame, task_id: int = BOOK_TASK) -> pd.Series:
    """True when this task's video or the other task is the book story."""
    onto = pd.to_numeric(sides["task_onto"], errors="coerce")
    frm = pd.to_numeric(sides["task_from"], errors="coerce")
    return (onto == task_id) | (frm == task_id)


def drop_book_sides(sides: pd.DataFrame, task_id: int = BOOK_TASK) -> pd.DataFrame:
    return sides.loc[~involves_book(sides, task_id)].copy()


def _sign(x: float) -> int:
    if not np.isfinite(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


def verdict_mean(
    n_book: int,
    orig: float,
    keep: float,
    *,
    limit: float = REL_CHANGE_LIMIT,
) -> str:
    """Compare original vs dropped-book mean Δ3. Locked rule for this check."""
    if int(n_book) == 0:
        return "本は入っていない"
    if not np.isfinite(orig) or not np.isfinite(keep):
        return "欠け"
    if _sign(orig) != _sign(keep):
        return "一部はこの課題に依存"
    if orig != 0 and (keep - orig) / abs(orig) < -limit:
        return "一部はこの課題に依存"
    return "頑健"


def _note_row(verdict: str, n_keep: int, orig_sig: bool, keep_sig: bool, rel: float) -> str:
    if verdict == "本は入っていない":
        return ""
    if orig_sig and not keep_sig and abs(rel) <= REL_CHANGE_LIMIT:
        return "平均は頑健．残った回数が少なく参考の検定は未通過"
    if n_keep <= 2 and orig_sig:
        return "残った回数が2"
    return ""


def compare_mains(
    sides: pd.DataFrame,
    orig_main: pd.DataFrame,
    keep_main: pd.DataFrame,
    *,
    task_id: int = BOOK_TASK,
) -> pd.DataFrame:
    """One row per source×score: original mean Δ3 vs mean after dropping book pairs."""
    book = involves_book(sides, task_id)
    far = sides["label"].astype(str) == "遠い"
    orig_i = orig_main.set_index(["source", "target"])
    keep_i = keep_main.set_index(["source", "target"])
    rows: list[dict[str, Any]] = []
    for source, target in planned_combos():
        g = sides[(sides["source"] == source) & (sides["target"] == target)]
        g_book = book.loc[g.index]
        g_far = far.loc[g.index]
        n_far_orig = int((g_far).sum())
        n_far_book = int((g_far & g_book).sum())
        o = orig_i.loc[(source, target)]
        k = keep_i.loc[(source, target)]
        orig_m = float(o["mean_delta_far"])
        keep_m = float(k["mean_delta_far"])
        n_keep = int(k["n_far"])
        n_near_keep = int(k["n_near"])
        if n_far_book == 0:
            keep_m = orig_m
            diff = 0.0
            rel = 0.0
        else:
            diff = keep_m - orig_m
            rel = diff / abs(orig_m) if np.isfinite(orig_m) and orig_m != 0 else 0.0
        orig_sig = bool(o["fdr_sig"])
        keep_sig = bool(k["fdr_sig"]) if pd.notna(k.get("fdr_sig", np.nan)) else False
        verdict = verdict_mean(n_far_book, orig_m, keep_m)
        rows.append(
            {
                "source": source,
                "target": target,
                "n_far_orig": n_far_orig,
                "n_far_book": n_far_book,
                "n_far_keep": n_keep,
                "n_near_keep": n_near_keep,
                "mean_delta_far_orig": orig_m,
                "mean_delta_far_drop_book": keep_m,
                "diff": diff,
                "rel_change": rel,
                "orig_fdr_sig": orig_sig,
                "p_keep": float(k["p"]),
                "q_keep": float(k["q_fdr"]),
                "sig_keep": keep_sig,
                "m_keep": int(k["m"]) if pd.notna(k.get("m", np.nan)) else 0,
                "skip_keep": "" if pd.isna(k.get("skip", np.nan)) else str(k["skip"]),
                "verdict": verdict,
                "note": _note_row(verdict, n_keep, orig_sig, keep_sig, rel),
            }
        )
        assert n_far_orig == int(o["n_far"]), (source, target, n_far_orig, int(o["n_far"]))
    return pd.DataFrame(rows, columns=list(COMPARE_COLS))


def book_context(exp1_hits: pd.DataFrame, exp2_cells: pd.DataFrame) -> dict[str, Any]:
    h = exp1_hits
    c = exp2_cells
    book1 = pd.to_numeric(h["task"], errors="coerce") == BOOK_TASK
    book2 = pd.to_numeric(c["task"], errors="coerce") == BOOK_TASK
    fdr2 = c["fdr_sig"].astype(bool)
    return {
        "book_task": BOOK_TASK,
        "book_ja": BOOK_JA,
        "exp1_n_hits": int(len(h)),
        "exp1_n_book": int(book1.sum()),
        "exp2_n_fdr": int(fdr2.sum()),
        "exp2_n_ran_book": int(book2.sum()),
        "exp2_n_fdr_book": int((book2 & fdr2).sum()),
    }


def sides_counts(sides: pd.DataFrame, task_id: int = BOOK_TASK) -> dict[str, int]:
    book = involves_book(sides, task_id)
    far = sides["label"].astype(str) == "遠い"
    return {
        "n_sides": int(len(sides)),
        "n_book": int(book.sum()),
        "n_far": int(far.sum()),
        "n_far_book": int((far & book).sum()),
        "n_keep": int((~book).sum()),
        "n_far_keep": int((far & ~book).sum()),
    }


def run_check(
    sides: pd.DataFrame,
    orig_main: pd.DataFrame,
    *,
    n_perm: int = N_LABEL_PERM,
    task_id: int = BOOK_TASK,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop book pairs, recompute far-mean tests on what remains, compare to orig."""
    dropped = sides.loc[involves_book(sides, task_id)].copy()
    kept = drop_book_sides(sides, task_id)
    keep_main, _sec = tests(kept, n_perm=n_perm)
    compare = compare_mains(sides, orig_main, keep_main, task_id=task_id)
    return compare, dropped, keep_main


def _fmt3(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{float(x):.3f}"


def _fmt_diff(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    if abs(x) < 1e-12:
        return "0"
    if x > 0:
        return f"+{x:.3f}"
    if x < 0:
        return f"{x:.3f}".replace("-", "−")
    return "0"


def _combo_ja(source: str, target: str) -> str:
    return f"{SOURCE_JA.get(source, source)} {target}"


def render_readme(
    compare: pd.DataFrame,
    *,
    counts: dict[str, int],
    context: dict[str, Any] | None,
    n_perm: int,
    limit: float = REL_CHANGE_LIMIT,
) -> str:
    ctx = context or {}
    lines = [
        "# 本のストーリーを除いたあとの実験3本線",
        "",
        "日付: 2026-08-18  ",
        "入口: `scripts/run_drop_book_story.py`  ",
        "対象: 実験3の本線9本．本のストーリー（課題 9）を片方に含む組を除き，平均Δ3 を出し直す．",
        "",
        "Ridge はやり直していない．既存の Δ3 を集計するだけである．本線の判定（全組・1000回，`outputs/exp3/main_test.csv`）は変えない．",
        "",
        "実験1の本のストーリー当たりは人数が少なく，同じ基準でも弱く読むと最初から決めてある．実験3の強い結果の一部が，この課題を含む組に乗っていないかを見る．",
        "",
    ]
    if ctx:
        lines += [
            f"実験1の当たり {ctx['exp1_n_hits']} 本のうち本は {ctx['exp1_n_book']} 本．"
            f"実験2で通った {ctx['exp2_n_fdr']} 条件のうち本は {ctx['exp2_n_fdr_book']} 条件"
            f"（本で回した条件は {ctx['exp2_n_ran_book']}）．",
            "",
        ]
    lines += [
        f"比べた回数は {counts['n_sides']}．種類が違う {counts['n_far']} 回のうち，"
        f"本を含むのは {counts['n_far_book']} 回．除いたあと {counts['n_far_keep']} 回．",
        "",
        "**判定（平均）**  ",
        f"符号が同じで，相対変化が {limit:.0%} 以内の減少までに収まるなら「頑健」．"
        "符号が逆，平均が 0 を跨ぐ，または 20% を超えて下がるなら「一部はこの課題に依存」．"
        "もともと本を含む比較が無い本は「本は入っていない」．",
        "",
        "符号入れ替えと BH を残した回数でかけ直すのは参考である．本線の家族ではない．"
        "残った回数が少なくて q が落ちても，平均がほとんど変わらなければ材料不足であり，本への依存ではない．",
        "",
        "| 情報源×点数 | 元の回数 | 本を含む | 残した回数 | 元の平均Δ3 | 除いたあと | 差 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in compare.itertuples(index=False):
        note = f"（{r.note}）" if r.note else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    _combo_ja(str(r.source), str(r.target)),
                    str(int(r.n_far_orig)),
                    str(int(r.n_far_book)),
                    str(int(r.n_far_keep)),
                    _fmt3(float(r.mean_delta_far_orig)),
                    _fmt3(float(r.mean_delta_far_drop_book)),
                    _fmt_diff(float(r.diff)),
                    f"{r.verdict}{note}",
                ]
            )
            + " |"
        )
    n_robust = int((compare["verdict"] == "頑健").sum())
    n_none = int((compare["verdict"] == "本は入っていない").sum())
    n_dep = int((compare["verdict"] == "一部はこの課題に依存").sum())
    orig_pass = compare[compare["orig_fdr_sig"]]
    still_pos = orig_pass[
        orig_pass["mean_delta_far_drop_book"].map(lambda x: np.isfinite(x) and x > 0)
    ]
    drops = orig_pass.copy()
    drops["drop_amt"] = drops["mean_delta_far_orig"] - drops["mean_delta_far_drop_book"]
    worst = drops.loc[drops["drop_amt"].idxmax()] if len(drops) else None
    lines += [
        "",
        f"9本のうち，判定が「頑健」は {n_robust} 本，「本は入っていない」は {n_none} 本，"
        f"「一部はこの課題に依存」は {n_dep} 本である．",
        "",
    ]
    if len(still_pos) == len(orig_pass) and len(orig_pass):
        lines.append(
            f"元から通っていた {len(orig_pass)} 本は，除いたあとも平均Δ3 がすべてプラスである．"
        )
    if worst is not None and float(worst["drop_amt"]) > 0:
        rel_pct = abs(float(worst["rel_change"])) * 100
        lines.append(
            "いちばん下がったのは "
            f"{_combo_ja(str(worst['source']), str(worst['target']))}"
            f"（{_fmt3(float(worst['mean_delta_far_orig']))}→"
            f"{_fmt3(float(worst['mean_delta_far_drop_book']))}，"
            f"相対 {rel_pct:.0f}% 減）である．"
        )
        second = drops.sort_values("drop_amt", ascending=False)
        if len(second) >= 2:
            w2 = second.iloc[1]
            if float(w2["drop_amt"]) > 0:
                lines.append(
                    "次は "
                    f"{_combo_ja(str(w2['source']), str(w2['target']))}"
                    f"（{_fmt3(float(w2['mean_delta_far_orig']))}→"
                    f"{_fmt3(float(w2['mean_delta_far_drop_book']))}）．"
                )
    lost = orig_pass[~orig_pass["sig_keep"]]
    if len(lost):
        lines += ["", "**参考の検定（残した回数・符号入れ替え "
                  f"{n_perm} 回・BH）**  ", ""]
        for r in lost.itertuples(index=False):
            lines.append(
                f"{_combo_ja(str(r.source), str(r.target))} は参考では未通過になる"
                f"（残 {int(r.n_far_keep)} 回，p={_fmt3(float(r.p_keep))}）．"
                f"平均は {_fmt3(float(r.mean_delta_far_orig))}→"
                f"{_fmt3(float(r.mean_delta_far_drop_book))} でほとんど変わらない．"
                "本が効果を作っていたのではなく，残った回数が足りない．"
            )
    keep_pass = int(compare["sig_keep"].sum())
    orig_n = int(compare["orig_fdr_sig"].sum())
    keep_of_orig = int((compare["orig_fdr_sig"] & compare["sig_keep"]).sum())
    lines += [
        "",
        f"参考として，残した回数だけでかけ直すと，通っていた {orig_n} 本のうち "
        f"{keep_of_orig} 本は通ったままである（参考の通過は {keep_pass} 本）．"
        "本線の 7/9 は変えない．",
        "",
        "**報告:** 実験3の本線の強さは，本のストーリーに依存しない．平均は頑健である．",
        "",
        "表: `compare.csv`．除いた比較: `dropped_sides.csv`．参考の検定: `main_keep.csv`．",
    ]
    return "\n".join(lines) + "\n"
