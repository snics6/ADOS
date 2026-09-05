#!/usr/bin/env python3
"""Generate outputs/sharing/*.html from current experiment CSVs."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/ADOS-exam")
OUT = ROOT / "outputs/sharing"
OUT.mkdir(parents=True, exist_ok=True)

TASK_JA = {
    1: "構成課題",
    2: "ごっこあそび",
    3: "共同注意",
    4: "実演",
    5: "絵の説明",
    6: "本のストーリーの説明",
    7: "自由遊び",
    8: "誕生日",
    9: "おやつ",
    10: "ルーティン",
}
TASK_SHORT = {
    1: "1 構成",
    2: "2 ごっこ",
    3: "3 共同注意",
    4: "4 実演",
    5: "5 絵",
    6: "6 本",
    7: "7 自由遊び",
    8: "8 誕生日",
    9: "9 おやつ",
    10: "10 ルーティン",
}
SRC_JA = {"child": "子ども", "examiner": "検査者", "dyad": "二人"}

CSS = r"""
:root {
  --bg: #f4f2ee; --panel: #fff; --ink: #1c1b19; --muted: #5c574f;
  --line: #ddd6cb; --accent: #0f5c6e; --sea: #d7eef2; --pos: #2f6b4f; --neg: #9b3d2e;
  --chip: #efeae2; --warn: #f3e6d4;
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: "IBM Plex Sans JP", "Hiragino Sans", "Noto Sans JP", sans-serif;
  background: var(--bg); color: var(--ink); line-height: 1.55;
}
a { color: var(--accent); }
.wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 64px; }
nav { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
nav a {
  text-decoration: none; background: var(--panel); border: 1px solid var(--line);
  padding: 6px 12px; border-radius: 6px; font-size: 13px; color: var(--ink);
}
nav a.active { background: var(--accent); color: #fff; border-color: var(--accent); }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.02em; }
.sub { color: var(--muted); margin: 0 0 18px; font-size: 14px; }
.callout {
  background: var(--sea); border-left: 4px solid var(--accent);
  padding: 12px 14px; margin-bottom: 16px; font-size: 14px;
}
.callout.warn { background: var(--warn); border-left-color: #8a5a00; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 22px; }
@media (max-width: 800px) { .stats { grid-template-columns: 1fr 1fr; } }
.stat { background: var(--panel); border: 1px solid var(--line); padding: 14px 16px; }
.stat b { display: block; font-size: 24px; font-variant-numeric: tabular-nums; }
.stat span { color: var(--muted); font-size: 12px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 22px; }
@media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } }
.card { background: var(--panel); border: 1px solid var(--line); padding: 14px 16px; }
.card h2 { font-size: 13px; margin: 0 0 10px; color: var(--muted); font-weight: 600; }
.filters { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 8px 0 14px; }
select {
  font: inherit; font-size: 13px; padding: 6px 10px; border: 1px solid var(--line);
  background: var(--panel); border-radius: 4px;
}
.count { color: var(--muted); font-size: 13px; }
h2.sec { font-size: 18px; margin: 28px 0 10px; }
.note { color: var(--muted); font-size: 13px; margin: 0 0 10px; }
.table-wrap { overflow: auto; border: 1px solid var(--line); background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 7px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th {
  position: sticky; top: 0; background: var(--chip); color: var(--muted);
  font-weight: 600; white-space: nowrap;
}
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.pos td:first-child { box-shadow: inset 3px 0 0 var(--pos); }
tr.neg td:first-child { box-shadow: inset 3px 0 0 var(--neg); }
tr.hit td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
.muted { color: var(--muted); }
.bar-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 12px; }
.bar-label { width: 84px; flex-shrink: 0; color: var(--muted); }
.bar-track { flex: 1; height: 14px; background: var(--chip); display: flex; overflow: hidden; }
.bar-seg { height: 100%; }
.legend { display: flex; gap: 12px; font-size: 12px; color: var(--muted); margin-top: 8px; flex-wrap: wrap; }
.swatch { display: inline-block; width: 10px; height: 10px; margin-right: 4px; vertical-align: middle; }
.pie-wrap { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
.footer { margin-top: 36px; color: var(--muted); font-size: 12px; }
code { font-size: 11px; background: var(--chip); padding: 1px 4px; }
.ol { margin: 0; padding-left: 1.2em; font-size: 14px; }
.ol li { margin: 4px 0; }
.fig { width: 100%; max-width: 100%; height: auto; border: 1px solid var(--line); background: #fff; }
tr.extra td:first-child { box-shadow: inset 3px 0 0 #8a5a00; }
.math-sec, .prose { background: var(--panel); border: 1px solid var(--line); padding: 18px 22px; margin: 12px 0 22px; font-size: 15px; line-height: 1.75; }
.math-sec h2, .prose h2 { font-size: 18px; margin: 28px 0 10px; color: var(--ink); font-weight: 600; }
.math-sec h2:first-child, .prose h2:first-child { margin-top: 0; }
.math-sec h3, .prose h3 { font-size: 16px; margin: 22px 0 8px; }
.math-sec p, .prose p { margin: 10px 0; }
.math-sec ol, .prose ol, .math-sec ul, .prose ul { margin: 8px 0 12px; padding-left: 1.4em; }
.math-sec li, .prose li { margin: 4px 0; }
.ex { background: var(--chip); padding: 12px 14px; margin: 14px 0; overflow-x: auto; }
.ex table { font-size: 13px; background: #fff; }
.katex-display { overflow-x: auto; overflow-y: hidden; padding: 6px 0; }
.katex { font-size: 1.08em; }
"""

JS_COMMON = r"""
const TASK_SHORT = {
  1:"1 構成",2:"2 ごっこ",3:"3 共同注意",4:"4 実演",5:"5 絵",
  6:"6 本",7:"7 自由遊び",8:"8 誕生日",9:"9 おやつ",10:"10 ルーティン"
};
const SRC_JA = {child:"子ども", examiner:"検査者", dyad:"二人"};
function fmtRho(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const s = Number(x).toFixed(3);
  return x > 0 ? "+"+s : s;
}
function fmtCI(lo, hi) {
  if (lo === null || hi === null) return "—";
  return "[" + fmtRho(lo) + ", " + fmtRho(hi) + "]";
}
function fmtP(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const n = Number(x);
  if (n < 0.001) return n.toExponential(1);
  return n.toFixed(3);
}
function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function stackedBars(el, cats, series) {
  const max = Math.max(1, ...cats.map((_,i)=>series.reduce((a,s)=>a+(s.data[i]||0),0)));
  const colors = ["#0f5c6e","#b85c38","#3d6b4f","#6b5b95"];
  el.innerHTML = cats.map((c,i)=>{
    const total = series.reduce((a,s)=>a+(s.data[i]||0),0);
    const segs = series.map((s,si)=>{
      const v = s.data[i]||0;
      const w = (v/max)*100;
      return `<div class="bar-seg" style="width:${w}%;background:${colors[si%colors.length]}" title="${esc(s.name)}: ${v}"></div>`;
    }).join("");
    return `<div class="bar-row"><div class="bar-label">${esc(c)}</div><div class="bar-track">${segs}</div><div class="num muted">${total}</div></div>`;
  }).join("") + `<div class="legend">${series.map((s,si)=>`<span><i class="swatch" style="background:${colors[si%colors.length]}"></i>${esc(s.name)}</span>`).join("")}</div>`;
}
function pieLegend(el, parts) {
  const total = parts.reduce((a,p)=>a+p.value,0) || 1;
  const colors = ["#0f5c6e","#b85c38","#3d6b4f","#6b5b95","#8a5a00"];
  let acc = 0;
  const stops = parts.map((p,i)=>{
    const a = acc; acc += (p.value/total)*100;
    return `${colors[i%colors.length]} ${a}% ${acc}%`;
  }).join(",");
  el.innerHTML = `<div class="pie-wrap">
    <div style="width:140px;height:140px;border-radius:50%;background:conic-gradient(${stops})"></div>
    <div>${parts.map((p,i)=>`<div><i class="swatch" style="background:${colors[i%colors.length]}"></i>${esc(p.label)} · <b>${p.value}</b></div>`).join("")}</div>
  </div>`;
}
"""


def nav(active: str) -> str:
    items = [
        ("index.html", "索引", "index"),
        ("exp1.html", "実験1", "exp1"),
        ("exp2.html", "実験2", "exp2"),
        ("exp3.html", "実験3", "exp3"),
        ("exp2_ridge.html", "実験2補助", "exp2_ridge"),
    ]
    bits = []
    for href, label, key in items:
        cls = "active" if key == active else ""
        bits.append(f'<a href="{href}" class="{cls}">{label}</a>')
    return "<nav>" + "".join(bits) + "</nav>"


def page(title: str, active: str, body: str, extra_js: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+JP:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"/>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{nav(active)}
{body}
<p class="footer">手動課題分割 · n=59 · 実験1（名簿）· 実験2（LOPO 校正）· 実験3（λ*, r, 2現象） · outputs/sharing/</p>
</div>
<script>
{JS_COMMON}
{extra_js}
</script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'\\\\[',right:'\\\\]',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}}],throwOnError:false,ignoredTags:['script','noscript','style','textarea','pre','code']}});"></script>
</body>
</html>
"""


def load_gloss() -> dict[str, tuple[str, str]]:
    gloss: dict[str, tuple[str, str]] = {}
    for cand in (
        ROOT / "outputs/share/exp1.html",
        ROOT / "outputs/sharing/exp1.html",
    ):
        if not cand.exists():
            continue
        html = cand.read_text(encoding="utf-8")
        m = re.search(r"const HITS = (\[.*?\]);", html, re.S)
        if not m:
            continue
        for h in json.loads(m.group(1)):
            gloss[h["feature"]] = (h.get("featureJa") or h["feature"], h.get("how") or "")
        break
    extra = {
        "txt_examiner_n_utt": ("検査者の発話数", "ASR発話セグメント数（検査者）"),
        "examiner_yaw_rot_per_min": ("検査者のヨー回転量", "1分あたりの左右首振りの総回転量"),
        "ges_child_fidgeting_per_min": ("子どものそわそわ", "姿勢イベント fidgeting の1分あたり回数"),
        "txt_child_n_utt": ("子どもの発話数", "ASR発話セグメント数（子ども）"),
        "examiner_turn_words_mean": ("検査者のターンあたり語数", "各発話ターンの語数の平均"),
        "child_yaw_rot_per_min": ("子どものヨー回転量", "1分あたりの左右首振りの総回転量"),
        "rsp_n_exam_turns": ("検査者ターン数", "課題内の検査者発話ターンの総数"),
        "win_child_speech_frac_delta": ("子ども発話割合の前後差", "後半−前半の発話時間割合"),
        "txt_child_question_frac": ("子どもの質問割合", "文末が？/か等の発話の割合（ASR）"),
        "examiner_roll_angacc_mean": ("検査者のロール角加速度の平均", "頭の傾き角の加速度の絶対値の平均"),
        "examiner_pitch_angacc_p90": ("検査者のピッチ角加速度の上側", "上下の頷き加速度の絶対値の90パーセンタイル"),
        "ges_child_hand_near_face_per_min": ("子どもの手–顔近接", "hand_near_face イベントの1分あたり回数"),
        "ges_child_leaning_away_per_min": ("子どもの体を引く", "leaning_away イベントの1分あたり回数"),
        "rsp_child_reply_frac": ("子どもの即返答率", "検査者ターンの直後2秒以内に子どもが続く割合"),
        "child_pitch_reversal_rate": ("子どものピッチ反転率", "上下方向の回転が符号を変える頻度"),
        "ges_examiner_hand_near_face_per_min": ("検査者の手–顔近接", "hand_near_face イベントの1分あたり回数"),
        "win_examiner_turn_rate_delta": ("検査者ターン率の前後差", "後半−前半の1分あたりターン数"),
        "examiner_pitch_reversal_rate": ("検査者のピッチ反転率", "上下方向の回転が符号を変える頻度"),
        "examiner_turn_rate_per_min": ("検査者のターン率", "1分あたりの発話ターン数"),
        "dyad_chain_max": ("やり取り鎖の最長", "課題内で最も長いやり取り鎖の長さ"),
        "dyad_silence_frac": ("沈黙の割合", "二人がどちらも発話していない時間の割合"),
        "dyad_turn_rate_per_min": ("二人のターン率", "1分あたりの話者交代回数"),
        "examiner_speech_frac": ("検査者の発話割合", "課題時間に占める検査者発話の割合"),
        "txt_examiner_chars_per_min": ("検査者の文字速度", "1分あたりのASR文字数（検査者）"),
        "win_child_turn_rate_delta": ("子どもターン率の前後差", "後半−前半の1分あたりターン数"),
        "txt_examiner_backchannel_frac": ("検査者のあいづち割合", "短いあいづちと判定した発話の割合"),
        "child_sway_y": ("子どもの上下の揺れ", "鼻キーポイントの縦位置の標準偏差"),
        "rsp_long_silence_frac": ("長い沈黙の割合", "検査者ターン後の間が2秒超の割合"),
        "examiner_roll_angvel_p90": ("検査者のロール角速度の上側", "頭の傾き速度の絶対値の90パーセンタイル"),
    }
    for k, v in extra.items():
        gloss.setdefault(k, v)
    return gloss


def ja_feat(name: str, gloss: dict[str, tuple[str, str]]) -> tuple[str, str]:
    return gloss.get(name, (name, ""))


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


def _f(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


GLOSS = load_gloss()

# ---------- exp1 ----------
hits = pd.read_csv(ROOT / "outputs/exp1/hits_main.csv")
meta1 = json.loads((ROOT / "outputs/exp1/meta.json").read_text())
rows1 = []
for r in hits.itertuples(index=False):
    ja, how = ja_feat(r.feature, GLOSS)
    rows1.append(
        {
            "task": int(r.task),
            "taskJa": r.task_ja,
            "target": r.target,
            "source": r.source,
            "feature": r.feature,
            "featureJa": ja,
            "how": how,
            "kind": "",
            "n": int(r.n),
            "rho": round(float(r.rho), 4),
            "p": float(r.p),
            "q": float(r.q_fdr),
            "stab": float(r.stability),
            "asr": bool(r.asr_risky),
            "m": int(r.m),
        }
    )

CATS = [TASK_SHORT[i] for i in range(1, 11)]


def count_by_task(df, pred):
    out = []
    for t in range(1, 11):
        sub = df[df["task"] == t]
        out.append(int(pred(sub).sum()) if len(sub) else 0)
    return out


sa = count_by_task(hits, lambda d: d["target"] == "SA")
rrb = count_by_task(hits, lambda d: d["target"] == "RRB")
css = count_by_task(hits, lambda d: d["target"] == "CSS")
pos = count_by_task(hits, lambda d: d["rho"] > 0)
neg = count_by_task(hits, lambda d: d["rho"] < 0)

body1 = f"""
<h1>実験1 · 単変量スクリーニング</h1>
<p class="sub">両側 Spearman · Stage A 安定度（同符号かつ |ρ|≥0.20）· BH は課題×得点 · n=59 · 手動課題分割</p>
<div class="callout">ステージA {meta1['n_stage_a']} → 候補 {meta1['n_candidates']} → FDR 当たり {meta1['n_main_fdr_hits']}（正 {(hits.rho>0).sum()} / 負 {(hits.rho<0).sum()}）。絵の説明は当たりなし。負の相関は本・自由遊びに多い。同じ59人での探索であり、独立確認ではない。</div>
<div class="stats">
  <div class="stat"><b>{meta1['n_main_fdr_hits']}</b><span>FDR 当たり</span></div>
  <div class="stat"><b>{int((hits.rho>0).sum())} / {int((hits.rho<0).sum())}</b><span>正 / 負</span></div>
  <div class="stat"><b>{meta1['n_candidates']}</b><span>候補（Stage A）</span></div>
  <div class="stat"><b>{meta1['n_stage_a']}</b><span>Stage A 本数</span></div>
</div>
<div class="grid2">
  <div class="card"><h2>課題ごとの当たり数（得点別）</h2><div id="barsTarget"></div></div>
  <div class="card"><h2>符号の内訳（課題別）</h2><div id="barsSign"></div></div>
</div>
<div class="grid2">
  <div class="card"><h2>得点</h2><div id="pieTarget"></div></div>
  <div class="card"><h2>情報源</h2><div id="pieSource"></div></div>
</div>
<h2 class="sec">手続き（実行した版）</h2>
<ol class="ol">
  <li>人×課題の特徴表を読む。体の大きさ補正はかけない。n&lt;8 の列は見ない。</li>
  <li>Stage A：重ならない半分を 200 回。両方 |ρ|≥0.20 かつ同符号なら加点。割合が安定度 S。</li>
  <li>閾値 τ は課題×得点ごとに、点数シャッフル 50 回の S の 95%点。本物の S を見る前にロック。</li>
  <li>候補は S≥τ（符号は問わない）。Stage B で両側置換 1000 回。BH は課題×得点。</li>
  <li>当たりは q&lt;0.05。ρ の符号は正でも負でもよい（現行コード）。</li>
</ol>
<h2 class="sec">フィルタ</h2>
<div class="filters">
  <select id="fTask"></select>
  <select id="fTgt"></select>
  <select id="fSrc"></select>
  <span class="count" id="fCount"></span>
</div>
<h2 class="sec">|ρ| 上位（フィルタ後）</h2>
<div class="table-wrap"><table><thead><tr>
<th>課題</th><th>得点</th><th>情報源</th><th>特徴</th><th>n</th><th>ρ</th><th>q</th><th>安定度</th>
</tr></thead><tbody id="topBody"></tbody></table></div>
<h2 class="sec">全当たり（フィルタ後）</h2>
<div class="table-wrap"><table><thead><tr>
<th>課題</th><th>得点</th><th>情報源</th><th>特徴</th><th>作り方</th><th>n</th><th>ρ</th><th>q</th><th>安定度</th><th>ASR</th>
</tr></thead><tbody id="allBody"></tbody></table></div>
<h2 class="sec">特徴の説明（フィルタに出たもの）</h2>
<p class="note">英語ID · 日本語名 · 算出の概要</p>
<div class="table-wrap"><table><thead><tr><th>英語ID</th><th>日本語名</th><th>作り方</th></tr></thead><tbody id="glossBody"></tbody></table></div>
"""

js1 = f"""
const HITS = {dumps(rows1)};
const CATS = {dumps(CATS)};
stackedBars(document.getElementById("barsTarget"), CATS, [
  {{name:"SA", data:{sa}}},
  {{name:"RRB", data:{rrb}}},
  {{name:"CSS", data:{css}}},
]);
stackedBars(document.getElementById("barsSign"), CATS, [
  {{name:"正", data:{pos}}},
  {{name:"負", data:{neg}}},
]);
pieLegend(document.getElementById("pieTarget"), [
  {{label:"SA", value:{int((hits.target=='SA').sum())}}},
  {{label:"CSS", value:{int((hits.target=='CSS').sum())}}},
  {{label:"RRB", value:{int((hits.target=='RRB').sum())}}},
]);
pieLegend(document.getElementById("pieSource"), [
  {{label:"子ども", value:{int((hits.source=='child').sum())}}},
  {{label:"検査者", value:{int((hits.source=='examiner').sum())}}},
  {{label:"二人", value:{int((hits.source=='dyad').sum())}}},
]);
const fTask=document.getElementById("fTask"), fTgt=document.getElementById("fTgt"), fSrc=document.getElementById("fSrc");
const taskIds = [...new Set(HITS.map(h=>h.task))].sort((a,b)=>a-b);
fTask.innerHTML = `<option value="all">課題: すべて</option>` + taskIds.map(t=>`<option value="${{t}}">${{TASK_SHORT[t]}}</option>`).join("");
fTgt.innerHTML = `<option value="all">得点: すべて</option><option value="SA">SA</option><option value="RRB">RRB</option><option value="CSS">CSS</option>`;
fSrc.innerHTML = `<option value="all">情報源: すべて</option><option value="child">子ども</option><option value="examiner">検査者</option><option value="dyad">二人</option>`;
function filtered() {{
  return HITS.filter(h =>
    (fTask.value==="all"||String(h.task)===fTask.value) &&
    (fTgt.value==="all"||h.target===fTgt.value) &&
    (fSrc.value==="all"||h.source===fSrc.value)
  );
}}
function render() {{
  const rows = filtered();
  document.getElementById("fCount").textContent = rows.length + " / " + HITS.length + " 件";
  const top = [...rows].sort((a,b)=>Math.abs(b.rho)-Math.abs(a.rho)).slice(0,12);
  document.getElementById("topBody").innerHTML = top.map(h=>`<tr class="${{h.rho>=0?'pos':'neg'}}"><td>${{esc(TASK_SHORT[h.task]||h.taskJa)}}</td><td>${{esc(h.target)}}</td><td>${{esc(SRC_JA[h.source]||h.source)}}</td><td>${{esc(h.featureJa)}}</td><td class="num">${{h.n}}</td><td class="num">${{fmtRho(h.rho)}}</td><td class="num">${{fmtP(h.q)}}</td><td class="num">${{h.stab.toFixed(2)}}</td></tr>`).join("");
  document.getElementById("allBody").innerHTML = rows.map(h=>`<tr class="${{h.rho>=0?'pos':'neg'}}"><td>${{esc(TASK_SHORT[h.task]||h.taskJa)}}</td><td>${{esc(h.target)}}</td><td>${{esc(SRC_JA[h.source]||h.source)}}</td><td>${{esc(h.featureJa)}}</td><td class="muted">${{esc(h.how)}}</td><td class="num">${{h.n}}</td><td class="num">${{fmtRho(h.rho)}}</td><td class="num">${{fmtP(h.q)}}</td><td class="num">${{h.stab.toFixed(2)}}</td><td>${{h.asr?"あり":""}}</td></tr>`).join("");
  const seen = new Map();
  rows.forEach(h=>{{ if(!seen.has(h.feature)) seen.set(h.feature, h); }});
  document.getElementById("glossBody").innerHTML = [...seen.values()].sort((a,b)=>a.feature.localeCompare(b.feature)).map(h=>`<tr><td><code>${{esc(h.feature)}}</code></td><td>${{esc(h.featureJa)}}</td><td>${{esc(h.how)}}</td></tr>`).join("");
}}
[fTask,fTgt,fSrc].forEach(el=>el.addEventListener("change", render));
render();
"""

(OUT / "exp1.html").write_text(page("実験1 · 手動課題分割", "exp1", body1, js1), encoding="utf-8")

# ---------- exp2 main ----------
c2 = pd.read_csv(ROOT / "outputs/exp2/cells.csv")
rows2 = []
for r in c2.itertuples(index=False):
    ja, how = ja_feat(str(r.cols), GLOSS)
    rows2.append(
        {
            "task": int(r.task),
            "taskJa": r.task_ja,
            "target": r.target,
            "source": r.source,
            "feature": r.cols,
            "featureJa": ja,
            "how": how,
            "kind": r.kinds,
            "n": int(r.n),
            "rho": round(float(r.rho), 4),
            "p": float(r.p),
            "q": float(r.q_fdr),
            "fdr": bool(r.fdr_sig),
            "m": int(r.m),
            "top1": round(float(r.top1_frac), 3) if pd.notna(r.top1_frac) else None,
        }
    )

body2 = f"""
<h1>実験2 · 本線（LOPO）</h1>
<p class="sub">学習 fold で Stage A → |ρ|最大の1本 → sign(訓練ρ)×x · 1人抜き · 外側置換 200 回 · BH は課題×得点</p>
<div class="callout">実験1で当たりがあった 41 セルだけ回す。候補は実験1の当たり列に固定せず、その情報源のカタログ全部。FDR 通過は <b>2 / 41</b>（自由遊び×検査者×SA、おやつ×検査者×SA）。同じ59人なので独立確認ではない。</div>
<div class="stats">
  <div class="stat"><b>41</b><span>実行セル</span></div>
  <div class="stat"><b>2</b><span>FDR 通過</span></div>
  <div class="stat"><b>{c2.rho.mean():.3f}</b><span>平均 ρ</span></div>
  <div class="stat"><b>{c2.rho.median():.3f}</b><span>中央 ρ</span></div>
</div>
<div class="grid2">
  <div class="card"><h2>課題ごとのセル数（FDR / その他）</h2><div id="barsFdr"></div></div>
  <div class="card"><h2>情報源</h2><div id="pieSource"></div></div>
</div>
<h2 class="sec">手続き</h2>
<ol class="ol">
  <li>セル = 実験1 FDR 当たりがある課題×情報源×得点。</li>
  <li>各人を1人外す。残りで S≥τ（τは実験1でロック）、通過者から訓練 |ρ| 最大の1本。</li>
  <li>外した人の予測は sign(訓練ρ)×その列。Ridge は使わない。標準化もしない。</li>
  <li>点数をシャッフルするたびに列選びからやり直す（n_perm=200）。</li>
  <li>BH は課題×得点。m はその課題・得点で実際に回った情報源の数。</li>
</ol>
<h2 class="sec">FDR 通過（2セル）</h2>
<div class="table-wrap"><table><thead><tr>
<th>課題</th><th>情報源</th><th>得点</th><th>ρ</th><th>p</th><th>q</th><th>特徴</th>
</tr></thead><tbody id="hitBody"></tbody></table></div>
<h2 class="sec">フィルタ</h2>
<div class="filters">
  <select id="fTask"></select>
  <select id="fTgt"></select>
  <select id="fSrc"></select>
  <select id="fFdr"></select>
  <span class="count" id="fCount"></span>
</div>
<h2 class="sec">全セル</h2>
<div class="table-wrap"><table><thead><tr>
<th>課題</th><th>得点</th><th>情報源</th><th>特徴</th><th>種類</th><th>n</th><th>ρ</th><th>p</th><th>q</th><th>FDR</th>
</tr></thead><tbody id="allBody"></tbody></table></div>
"""

fdr_by = count_by_task(c2, lambda d: d["fdr_sig"])
oth_by = count_by_task(c2, lambda d: ~d["fdr_sig"])

js2 = f"""
const CELLS = {dumps(rows2)};
const CATS = {dumps(CATS)};
stackedBars(document.getElementById("barsFdr"), CATS, [
  {{name:"FDR通過", data:{fdr_by}}},
  {{name:"未通過", data:{oth_by}}},
]);
pieLegend(document.getElementById("pieSource"), [
  {{label:"子ども", value:{int((c2.source=='child').sum())}}},
  {{label:"検査者", value:{int((c2.source=='examiner').sum())}}},
  {{label:"二人", value:{int((c2.source=='dyad').sum())}}},
]);
document.getElementById("hitBody").innerHTML = CELLS.filter(h=>h.fdr).map(h=>
  `<tr class="hit"><td>${{esc(TASK_SHORT[h.task])}}</td><td>${{esc(SRC_JA[h.source])}}</td><td>${{esc(h.target)}}</td><td class="num">${{fmtRho(h.rho)}}</td><td class="num">${{fmtP(h.p)}}</td><td class="num">${{fmtP(h.q)}}</td><td>${{esc(h.featureJa)}} <code>${{esc(h.feature)}}</code></td></tr>`
).join("");
const fTask=document.getElementById("fTask"), fTgt=document.getElementById("fTgt"), fSrc=document.getElementById("fSrc"), fFdr=document.getElementById("fFdr");
const taskIds = [...new Set(CELLS.map(h=>h.task))].sort((a,b)=>a-b);
fTask.innerHTML = `<option value="all">課題: すべて</option>` + taskIds.map(t=>`<option value="${{t}}">${{TASK_SHORT[t]}}</option>`).join("");
fTgt.innerHTML = `<option value="all">得点: すべて</option><option value="SA">SA</option><option value="RRB">RRB</option><option value="CSS">CSS</option>`;
fSrc.innerHTML = `<option value="all">情報源: すべて</option><option value="child">子ども</option><option value="examiner">検査者</option><option value="dyad">二人</option>`;
fFdr.innerHTML = `<option value="all">FDR: すべて</option><option value="yes">通過のみ</option><option value="no">未通過</option>`;
function filtered() {{
  return CELLS.filter(h =>
    (fTask.value==="all"||String(h.task)===fTask.value) &&
    (fTgt.value==="all"||h.target===fTgt.value) &&
    (fSrc.value==="all"||h.source===fSrc.value) &&
    (fFdr.value==="all"|| (fFdr.value==="yes"?h.fdr:!h.fdr))
  );
}}
function render() {{
  const rows = filtered();
  document.getElementById("fCount").textContent = rows.length + " / " + CELLS.length + " 件";
  document.getElementById("allBody").innerHTML = rows.map(h=>`<tr class="${{h.fdr?'hit':(h.rho>=0?'pos':'neg')}}"><td>${{esc(TASK_SHORT[h.task])}}</td><td>${{esc(h.target)}}</td><td>${{esc(SRC_JA[h.source])}}</td><td>${{esc(h.featureJa)}}</td><td>${{esc(h.kind||"")}}</td><td class="num">${{h.n}}</td><td class="num">${{fmtRho(h.rho)}}</td><td class="num">${{fmtP(h.p)}}</td><td class="num">${{fmtP(h.q)}}</td><td>${{h.fdr?"通過":""}}</td></tr>`).join("");
}}
[fTask,fTgt,fSrc,fFdr].forEach(el=>el.addEventListener("change", render));
render();
"""

(OUT / "exp2.html").write_text(page("実験2本線 · LOPO", "exp2", body2, js2), encoding="utf-8")

# ---------- exp2 ridge aux ----------
c3 = pd.read_csv(ROOT / "outputs/exp2_ridge/cells.csv")
rows3 = []
for r in c3.itertuples(index=False):
    feat = str(r.cols).split("|")[0]
    ja, how = ja_feat(feat, GLOSS)
    rows3.append(
        {
            "task": int(r.task),
            "taskJa": r.task_ja,
            "target": r.target,
            "source": r.source,
            "feature": str(r.cols),
            "featureJa": ja,
            "how": how,
            "kind": r.kinds,
            "k": int(r.k),
            "n": int(r.n),
            "rho": round(float(r.rho), 4),
            "p": float(r.p),
            "q": float(r.q_fdr),
            "fdr": bool(r.fdr_sig),
            "m": int(r.m),
        }
    )

body3 = f"""
<h1>実験2補助 · 実験1の当たりをそのまま Ridge</h1>
<p class="sub">本線ではない。実験1で残った1本または2本を fold 内で選び直さず Ridge · 置換 1000 回 · BH は課題×得点</p>
<div class="callout warn">これは探索的な確認用。列は実験1の当たりそのものなので、実験1と数字が近くなりやすい。論文の主結果は実験3（対比）。この補助の FDR 37 から切った旧い移植・プールは本線にしない。</div>
<div class="stats">
  <div class="stat"><b>41</b><span>実行セル</span></div>
  <div class="stat"><b>37</b><span>FDR 通過</span></div>
  <div class="stat"><b>{int((c3.k==1).sum())} / {int((c3.k==2).sum())}</b><span>k=1 / k=2</span></div>
  <div class="stat"><b>49</b><span>スキップ（当たり0など）</span></div>
</div>
<div class="grid2">
  <div class="card"><h2>課題ごとのセル数（FDR / その他）</h2><div id="barsFdr"></div></div>
  <div class="card"><h2>情報源</h2><div id="pieSource"></div></div>
</div>
<h2 class="sec">手続き</h2>
<ol class="ol">
  <li>実験1当たりが1本ならその1本、2本以上なら上位2本（並べは q→ρ→安定度→名前）。</li>
  <li>両方測れている人だけ。n&lt;12 は回さない。</li>
  <li>学習側だけで標準化し RidgeCV（α は logspace）。評価側は transform のみ。</li>
  <li>fold 数は n に合わせる（≥40 なら 5×10 など）。層は得点分位、可能なら早産印も。</li>
  <li>ρ は fold 平均の Spearman。p は点数シャッフル 1000 回（片側・大きい方）。</li>
</ol>
<h2 class="sec">フィルタ</h2>
<div class="filters">
  <select id="fTask"></select>
  <select id="fTgt"></select>
  <select id="fSrc"></select>
  <select id="fFdr"></select>
  <span class="count" id="fCount"></span>
</div>
<h2 class="sec">全セル</h2>
<div class="table-wrap"><table><thead><tr>
<th>課題</th><th>得点</th><th>情報源</th><th>k</th><th>特徴</th><th>n</th><th>ρ</th><th>p</th><th>q</th><th>FDR</th>
</tr></thead><tbody id="allBody"></tbody></table></div>
"""

fdr_by3 = count_by_task(c3, lambda d: d["fdr_sig"])
oth_by3 = count_by_task(c3, lambda d: ~d["fdr_sig"])

js3 = f"""
const CELLS = {dumps(rows3)};
const CATS = {dumps(CATS)};
stackedBars(document.getElementById("barsFdr"), CATS, [
  {{name:"FDR通過", data:{fdr_by3}}},
  {{name:"未通過", data:{oth_by3}}},
]);
pieLegend(document.getElementById("pieSource"), [
  {{label:"子ども", value:{int((c3.source=='child').sum())}}},
  {{label:"検査者", value:{int((c3.source=='examiner').sum())}}},
  {{label:"二人", value:{int((c3.source=='dyad').sum())}}},
]);
const fTask=document.getElementById("fTask"), fTgt=document.getElementById("fTgt"), fSrc=document.getElementById("fSrc"), fFdr=document.getElementById("fFdr");
const taskIds = [...new Set(CELLS.map(h=>h.task))].sort((a,b)=>a-b);
fTask.innerHTML = `<option value="all">課題: すべて</option>` + taskIds.map(t=>`<option value="${{t}}">${{TASK_SHORT[t]}}</option>`).join("");
fTgt.innerHTML = `<option value="all">得点: すべて</option><option value="SA">SA</option><option value="RRB">RRB</option><option value="CSS">CSS</option>`;
fSrc.innerHTML = `<option value="all">情報源: すべて</option><option value="child">子ども</option><option value="examiner">検査者</option><option value="dyad">二人</option>`;
fFdr.innerHTML = `<option value="all">FDR: すべて</option><option value="yes">通過のみ</option><option value="no">未通過</option>`;
function filtered() {{
  return CELLS.filter(h =>
    (fTask.value==="all"||String(h.task)===fTask.value) &&
    (fTgt.value==="all"||h.target===fTgt.value) &&
    (fSrc.value==="all"||h.source===fSrc.value) &&
    (fFdr.value==="all"|| (fFdr.value==="yes"?h.fdr:!h.fdr))
  );
}}
function render() {{
  const rows = filtered();
  document.getElementById("fCount").textContent = rows.length + " / " + CELLS.length + " 件";
  document.getElementById("allBody").innerHTML = rows.map(h=>`<tr class="${{h.fdr?'hit':(h.rho>=0?'pos':'neg')}}"><td>${{esc(TASK_SHORT[h.task])}}</td><td>${{esc(h.target)}}</td><td>${{esc(SRC_JA[h.source])}}</td><td class="num">${{h.k}}</td><td>${{esc(h.featureJa)}} <code>${{esc(h.feature)}}</code></td><td class="num">${{h.n}}</td><td class="num">${{fmtRho(h.rho)}}</td><td class="num">${{fmtP(h.p)}}</td><td class="num">${{fmtP(h.q)}}</td><td>${{h.fdr?"通過":""}}</td></tr>`).join("");
}}
[fTask,fTgt,fSrc,fFdr].forEach(el=>el.addEventListener("change", render));
render();
"""

(OUT / "exp2_ridge.html").write_text(
    page("実験2補助 · Ridge", "exp2_ridge", body3, js3), encoding="utf-8"
)

# ---------- exp3 pairs ----------
p21 = pd.read_csv(ROOT / "outputs/hetero/pairs_21.csv")
s10 = pd.read_csv(ROOT / "outputs/hetero/signflip_10.csv")
v10_path = ROOT / "outputs/hetero/pool_verify_10.csv"
v10 = pd.read_csv(v10_path) if v10_path.exists() else pd.DataFrame()

def _opt_csv(name: str) -> pd.DataFrame:
    p = ROOT / "outputs/hetero" / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

qr = _opt_csv("q_ranks.csv")
dur = _opt_csv("duration_by_task.csv")
part = _opt_csv("duration_partial.csv")
init = _opt_csv("initiator_rho.csv")
init_n = _opt_csv("initiator_counts.csv")
dur_y = _opt_csv("duration_y.csv")
maxs = _opt_csv("maxstat_3hit.csv")

for fig_name in (
    "fig_exp3_main.png",
    "fig_r_hist.png",
    "fig_lambda_r.png",
    "fig_ci_forest.png",
    "fig_duration.png",
    "fig_duration_y.png",
    "fig_r_vs_dur.png",
    "fig_q_ranks.png",
    "fig_initiator.png",
    "fig_lambda_unsel.png",
    "fig_lambda_perm.png",
):
    src = ROOT / "outputs/hetero" / fig_name
    if src.exists():
        shutil.copy(src, OUT / fig_name)
stale_eq3 = OUT / "fig_eq3_verify.png"
if stale_eq3.exists():
    stale_eq3.unlink()

def _phen(feat: str, pos_ja: str, neg_ja: str) -> str:
    if str(feat).startswith("win_"):
        return "発話ドリフト"
    return "共同注意 vs 自由遊び"

rows21 = []
for i, r in enumerate(p21.itertuples(index=False)):
    ja, how = ja_feat(str(r.feature), GLOSS)
    lam = float(p21["lambda"].iloc[i])
    rows21.append(
        {
            "feature": r.feature,
            "featureJa": ja,
            "how": how,
            "target": r.target,
            "source": r.source,
            "taskA": int(r.task_a),
            "taskB": int(r.task_b),
            "taskAJa": r.task_a_ja,
            "taskBJa": r.task_b_ja,
            "n": int(r.n_cap),
            "rhoA": round(float(r.rho_cap_a), 4),
            "rhoB": round(float(r.rho_cap_b), 4),
            "r": round(float(r.r_spear), 3),
            "lambda": None if not np.isfinite(lam) else round(lam, 3),
            "flip": bool(r.sign_flip_cap),
            "poolWorse": bool(r.pool_worse),
            "dz": round(float(r.delta_rho), 3) if hasattr(r, "delta_rho") and pd.notna(r.delta_rho) else None,
            "dzLo": _f(r.ci_rho_lo) if hasattr(r, "ci_rho_lo") else None,
            "dzHi": _f(r.ci_rho_hi) if hasattr(r, "ci_rho_hi") else None,
            "gainObs": (
                None
                if not hasattr(r, "p1_gain_obs") or pd.isna(r.p1_gain_obs)
                else round(float(r.p1_gain_obs), 4)
            ),
        }
    )

rows10 = []
for _, row in s10.iterrows():
    ja, how = ja_feat(str(row["feature"]), GLOSS)
    qv = row["q_fdr_21"]
    rows10.append(
        {
            "feature": row["feature"],
            "featureJa": ja,
            "target": row["target"],
            "source": row["source"],
            "posJa": row["task_pos_ja"],
            "negJa": row["task_neg_ja"],
            "n": int(row["n_cap"]),
            "rhoPos": round(float(row["rho_cap_pos"]), 4),
            "rhoNeg": round(float(row["rho_cap_neg"]), 4),
            "r": round(float(row["r_spear"]), 3) if pd.notna(row["r_spear"]) else None,
            "p": _f(row["p_boot"]),
            "pHw": _f(row["p_hw"]),
            "q": None if pd.isna(qv) else _f(qv),
            "in21": bool(row["in_pairs_21"]),
            "fdr": bool(row["fdr_sig_21"]) if pd.notna(row["fdr_sig_21"]) else False,
            "phen": _phen(str(row["feature"]), str(row["task_pos_ja"]), str(row["task_neg_ja"])),
        }
    )

n_flip = int(p21["sign_flip_cap"].sum())
n_flip_sig = int(((p21["sign_flip_cap"]) & (p21["fdr_sig"])).sum())
n_same_sig = int((~p21["sign_flip_cap"] & p21["fdr_sig"]).sum())
n_worse = int(p21["pool_worse"].sum())
r_flip = p21.loc[p21["sign_flip_cap"], "r_spear"]
r_flip_med = float(r_flip.median()) if len(r_flip) else float("nan")
r_same = p21.loc[~p21["sign_flip_cap"], "r_spear"]
r_same_med = float(r_same.median()) if len(r_same) else float("nan")
n_vz = int(v10["rho_z_covers_0"].sum()) if not v10.empty else 0
n_vp = int(v10["rho_pos_excludes_0"].sum()) if not v10.empty else 0
n_both = int((v10["rho_z_covers_0"] & v10["rho_pos_excludes_0"]).sum()) if not v10.empty else 0
w_plus = float((v10["rho_pos_hi"] - v10["rho_pos_lo"]).median()) if not v10.empty else float("nan")
w_z = float((v10["rho_z_hi"] - v10["rho_z_lo"]).median()) if not v10.empty else float("nan")
n_part_ok = int(part["sign_survives"].sum()) if not part.empty else 0
fid = p21[(p21["feature"] == "ges_child_fidgeting_per_min") & (p21["target"] == "CSS")]
fid_r = float(fid["r_spear"].iloc[0]) if not fid.empty else float("nan")
fid_l = float(fid["lambda"].iloc[0]) if not fid.empty else float("nan")
fid_ls = float(fid["lambda_star"].iloc[0]) if not fid.empty else float("nan")
meta_f = {}
mp = ROOT / "outputs/hetero/findings_meta.json"
if mp.exists():
    meta_f = json.loads(mp.read_text(encoding="utf-8"))
r_all_med = float(meta_f.get("r_median") or float("nan"))
r_all_q25 = float(meta_f.get("r_q25") or float("nan"))
r_all_q75 = float(meta_f.get("r_q75") or float("nan"))
lam_med = float(meta_f.get("lambda_star_at_median_r") or float("nan"))
lam_q25 = float(meta_f.get("lambda_star_at_q25") or float("nan"))
lam_q75 = float(meta_f.get("lambda_star_at_q75") or float("nan"))
ceil_r = float(meta_f.get("reliability_ceiling_sqrt_r") or float("nan"))
n_r_valid = int(meta_f.get("n_pairs_valid") or 0)
n_r_nom = int(meta_f.get("n_pairs_nominal") or 0)
frac_low = float(meta_f.get("frac_r_lt_0.2") or float("nan"))
n_p05 = int(meta_f.get("n_p_lt_05") or 0)
n_fam = int(meta_f.get("n_families") or 0)
exp_p05 = float(meta_f.get("expected_p_lt_05") or float("nan"))
lam_u = meta_f.get("lambda_unselected") or {}
n_lam = int(lam_u.get("n_cells") or 0)
frac_worse_u = float(lam_u.get("frac_pool_worse") or float("nan"))
frac_worse_same = float(lam_u.get("frac_pool_worse_same_sign") or float("nan"))
n_lam_same = int(lam_u.get("n_same_sign") or 0)
lam_p = meta_f.get("lambda_perm") or {}
lam_null = float(lam_p.get("null_median") or float("nan"))
lam_null95 = float(lam_p.get("null_q95") or float("nan"))
lam_pperm = float(lam_p.get("p_perm") or float("nan"))
lam_null_same = float(lam_p.get("null_same_median") or float("nan"))
q_ecdf = meta_f.get("ecdf") or []
def _ecdf(thr: float) -> dict:
    for e in q_ecdf:
        if abs(float(e.get("threshold", -1)) - thr) < 1e-9:
            return e
    return {}
e05, e10, e20 = _ecdf(0.05), _ecdf(0.10), _ecdf(0.20)
init_bin_meta = meta_f.get("initiator_binary") or []
dur_ym = meta_f.get("duration_y") or {}
rvd = meta_f.get("r_vs_duration") or {}
rvd_rho = float(rvd.get("spearman_r_vs_min_dur") or float("nan"))
rvd_pair = float(rvd.get("pair_level_spearman") or float("nan"))
rvd_bins = rvd.get("bin_medians") or []
rvd_bin_txt = ", ".join(f"{float(x):.2f}" for x in rvd_bins)

rows_q = []
for _, row in qr.iterrows():
    rows_q.append(
        {
            "feature": row["feature"],
            "target": row["target"],
            "phen": {"ja_free": "共同注意 vs 自由遊び", "speech_drift": "発話ドリフト", "other": "そわそわ"}.get(str(row.get("phenomenon", "")), str(row.get("phenomenon", ""))),
            "p": _f(row["p"]),
            "rank": int(row["p_rank"]),
            "nFam": int(row["n_fam"]),
        }
    )
rows_dur = []
for _, row in dur.iterrows():
    rows_dur.append(
        {
            "task": int(row["task"]),
            "ja": row["task_ja"],
            "n": int(row["n"]),
            "median": round(float(row["median_sec"]), 1),
            "q25": round(float(row["q25_sec"]), 1) if "q25_sec" in row and pd.notna(row["q25_sec"]) else None,
            "q75": round(float(row["q75_sec"]), 1) if "q75_sec" in row and pd.notna(row["q75_sec"]) else None,
            "mean": round(float(row["mean_sec"]), 1),
            "min": round(float(row["min_sec"]), 1),
            "max": round(float(row["max_sec"]), 1),
        }
    )
rows_part = []
for _, row in part.iterrows():
    ja, _how = ja_feat(str(row["feature"]), GLOSS)
    rows_part.append(
        {
            "feature": row["feature"],
            "featureJa": ja,
            "target": row["target"],
            "side": row["side"],
            "ja": row["task_ja"],
            "n": int(row["n"]),
            "rho": round(float(row["rho"]), 4),
            "rhoP": round(float(row["rho_partial"]), 4) if pd.notna(row["rho_partial"]) else None,
            "rhoXd": round(float(row["rho_x_duration"]), 3) if pd.notna(row["rho_x_duration"]) else None,
            "ok": bool(row["sign_survives"]),
            "rate": bool(row["rate_normalized"]) if "rate_normalized" in row.index and pd.notna(row["rate_normalized"]) else None,
        }
    )
rows_init = []
for _, row in init.iterrows():
    if "ge4_per_min" not in str(row["feature"]):
        continue
    rows_init.append(
        {
            "taskJa": row["task_ja"],
            "target": row["target"],
            "feature": row["feature"],
            "who": row["who"],
            "n": int(row["n"]),
            "rho": round(float(row["rho"]), 3) if pd.notna(row["rho"]) else None,
            "lo": _f(row["rho_lo"]) if "rho_lo" in row.index else None,
            "hi": _f(row["rho_hi"]) if "rho_hi" in row.index else None,
            "rhoP": round(float(row["rho_partial_duration"]), 3) if pd.notna(row["rho_partial_duration"]) else None,
        }
    )
rows_init_n = []
for _, row in init_n.iterrows():
    rows_init_n.append(
        {
            "taskJa": row["task_ja"],
            "who": row["who"],
            "n": int(row["n_sessions"]),
            "medN": round(float(row["median_n_chains"]), 1),
            "meanN": round(float(row["mean_n_chains"]), 1),
            "medGe4": round(float(row["median_n_ge4"]), 2),
            "meanGe4": round(float(row["mean_n_ge4"]), 2),
            "zero": round(100 * float(row["frac_zero_ge4"]), 0) if pd.notna(row["frac_zero_ge4"]) else None,
        }
    )
rows_dur_y = []
if not dur_y.empty:
    for tid, g in dur_y.groupby("task"):
        rec = {"task": int(tid), "ja": str(g["task_ja"].iloc[0])}
        for _, row in g.iterrows():
            rec[str(row["target"])] = round(float(row["rho"]), 2)
            rec["n"] = int(row["n"])
        rows_dur_y.append(rec)
rows_ms = []
for _, row in maxs.iterrows():
    rows_ms.append(
        {
            "feature": row["feature"],
            "target": row["target"],
            "nHit": int(row["n_hit_tasks"]),
            "nCand": int(row["n_cand_pairs"]),
            "p": _f(row["p_maxstat"]),
        }
    )

rows_v = []
if not v10.empty:
    for _, row in v10.iterrows():
        ja, _how = ja_feat(str(row["feature"]), GLOSS)
        rows_v.append(
            {
                "feature": row["feature"],
                "featureJa": ja,
                "target": row["target"],
                "posJa": row["task_pos_ja"],
                "negJa": row["task_neg_ja"],
                "n": int(row["n_cap"]),
                "rhoPos": round(float(row["rho_pos"]), 4),
                "rhoPosLo": _f(row["rho_pos_lo"]),
                "rhoPosHi": _f(row["rho_pos_hi"]),
                "posEx0": bool(row["rho_pos_excludes_0"]),
                "rhoNeg": round(float(row["rho_neg"]), 4),
                "r": round(float(row["r_spear"]), 3),
                "pred": round(float(row["rho_z_pred"]), 4),
                "obs": round(float(row["rho_z_obs"]), 4),
                "zLo": _f(row["rho_z_lo"]),
                "zHi": _f(row["rho_z_hi"]),
                "zCov0": bool(row["rho_z_covers_0"]),
                "absErr": _f(row["abs_err"]),
            }
        )

EXP3_PROSE = r"""
<div class="prose">
<h2>問い</h2>
<p>ADOS-2 Module 2 は 10 個の課題を順に行う。子どもは同じ1人でも、課題が変わると話し方と動き方が変わる。自動評価の多くは課題を区別せず、同じ名前の行動の数値を課題のあいだで平均する。これをプールと呼ぶ。</p>
<p>実験3の問い: 同じ名前の数値を違う課題のあいだで平均すると、重症度との関係は強くなるのか弱くなるのか。弱くなるとすれば、測り損ないなのか、課題が別のものを引き出しているのか。</p>

<h2>何を仮定したか</h2>
<ol>
<li>59人は実験1と同じコホートである。重症度（SA / RRB / CSS）は人に1本で、課題では変わらない。</li>
<li>課題区間は人手の <code>data/task_segments.json</code> である。</li>
<li>「一緒に動く」は、値を順位に直したあとの相関 \(\rho\)（次節）である。</li>
<li>プールは、2課題の数値を先に順位にしてから平均することである。生の値の平均ではない。</li>
<li>2課題を比べるときは、両方で数値が有限な人だけを使う。</li>
<li>21対は実験1の当たりから切っている。差が0かどうかの検定は結果にしない。</li>
</ol>
<p>置かないこと: 10課題が同じ特性を測っている、プールは常に有害、符号反転の母集団割合。</p>

<h2>「一緒に動く」の測り方</h2>
<p>課題Aの数値 \(x\) と重症度 \(y\) がある。生の値のまま掛け算すると単位が違う。先に順位に直す。いちばん小さい値を1、次を2、…とする。同じ値は、占める順位の平均を与える。</p>
<div class="ex">
<p>5人の例。発話が多い人ほど SA が高い。</p>
<table><thead><tr><th>人</th><th>発話の割合</th><th>発話の順位</th><th>SA</th><th>SA の順位</th></tr></thead>
<tbody>
<tr><td>A</td><td class="num">0.10</td><td class="num">1</td><td class="num">2</td><td class="num">1</td></tr>
<tr><td>C</td><td class="num">0.20</td><td class="num">2</td><td class="num">4</td><td class="num">2</td></tr>
<tr><td>E</td><td class="num">0.30</td><td class="num">3</td><td class="num">6</td><td class="num">3</td></tr>
<tr><td>B</td><td class="num">0.40</td><td class="num">4</td><td class="num">8</td><td class="num">4</td></tr>
<tr><td>D</td><td class="num">0.50</td><td class="num">5</td><td class="num">10</td><td class="num">5</td></tr>
</tbody></table>
<p>順位は完全に同じ並びである。</p>
</div>
<p>順位から平均を引いた列を \(u_i\) と \(v_i\) とする。対応する項を掛けて足し、各列の大きさで割る。</p>
$$
\rho=\frac{\sum u_i v_i}{\sqrt{(\sum u_i^2)(\sum v_i^2)}}.
$$
<p>\(\rho=1\) は順位が完全に同じ。\(\rho=-1\) は完全に逆。\(\rho=0\) は一方を知っても他方が読めない。範囲は \(-1\) から \(1\)。</p>
<p>実験3では同じ人の集合の上で3本を出す。</p>
$$
\rho_A=\rho(x_A,y),\qquad \rho_B=\rho(x_B,y),\qquad r=\rho(x_A,x_B).
$$
<p>\(r\) には重症度を使わない。同じ人が2つの課題で似た順位になるかどうかだけである。\(r\) が高い: 課題Aで大きい人は課題Bでも大きい。\(r\) が0に近い: 課題が変わると並びが入れ替わる。</p>

<h2>プールしたあと \(\rho\) がどうなるか</h2>
<p>人 \(i\) について、2課題の順位の平均を取る。</p>
$$
z_i=\frac{R(x_A)_i+R(x_B)_i}{2}.
$$
<p>この \(z\) と重症度の順位との \(\rho\) を \(\rho_z\) とする。定義から次が必ず成り立つ（人数を含まない）。</p>
$$
\rho_z=\frac{\rho_A+\rho_B}{\sqrt{2(1+r)}}.
$$
<p>導出: 平均を引いた順位列を \(U\)（課題A）、\(V\)（課題B）、\(W\)（重症度）、長さ \(s=\lVert U\rVert=\lVert V\rVert\) とする。\(\rho_A=(\sum U_i W_i)/(s\lVert W\rVert)\) など。プールは \(Z=(U+V)/2\)。分子は \(\frac12(\sum U_i W_i+\sum V_i W_i)\)。長さは \(\lVert U+V\rVert^2=2s^2(1+r)\)。割ると上の式になる。</p>
<p>\(r=1\)（2本が同じ並び）なら \(\rho_z=\rho_A\)。同じものを2回足しても増えない。\(r=0\) なら分母は \(\sqrt{2}\approx 1.414\)。</p>
<p>課題Bが無情報（\(\rho_B=0,\ r=0\)）なら \(\rho_z=\rho_A/\sqrt{2}\approx 0.707\,\rho_A\)。強い課題だけの関係の約 29% が消える。</p>
<p>2課題のうち \(\lvert\rho\rvert\) が大きいほうを A とし \(\rho_A \gt 0\) にそろえる。比 \(\lambda=\rho_B/\rho_A\)。プールが強い課題より得なのは \(\rho_z \gt \rho_A\) のときで、これは</p>
$$
\lambda \gt \lambda^*(r)=\sqrt{2(1+r)}-1
$$
<p>と同値である。\(r=0\) なら \(\lambda^*=0.414\)。\(r=0.25\) なら \(\lambda^*=0.580\)。弱い側が強い側の信号の 41%〜58% を残していないと、平均は損である。</p>
<p>向きが逆（\(\rho_B=-\rho_A,\ r=0\)）なら分子は0で \(\rho_z=0\)。各課題では重症度と関係しているのに、平均すると消える。</p>
<p>同じ強さ（\(\lambda=1\)）ならプールは単独より \(\sqrt{2/(1+r)}\) 倍良い。\(r=0.25\) なら約 26% 増。これは「無作為に選んだ1課題」との比較である。より強い課題と比べると、\(\lambda \lt \lambda^*\) ならプールは負ける。主張は「平均するな」ではなく「課題を潰すな」。</p>

<h2>何をしたか</h2>
<ol>
<li>67特徴 × 45課題対 = 3015 について、重症度を使わず \(r\) を出した（選択なし）。</li>
<li>符号が逆の10組で、実測 \(\rho_z\) と式の右辺を突き合わせ、人の復元抽出で区間を付けた。</li>
<li>実験1の当たりから切った21対の \(\rho_A,\rho_B\) と区間を出した。検定はしない。</li>
<li>やり取りの鎖を、検査者開始と子ども開始に分けた。</li>
<li>区間の長さで符号が消えるかを見た。</li>
</ol>

<h2>何が分かったか</h2>
<p><strong>同名でも課題が変わると並びはあまり一致しない。</strong> 3015対の \(r\) の中央は 0.25（四分位 0.10–0.41）。この位置では \(\lambda^*\approx 0.58\)。弱い課題が強い課題の半分から 3分の2 を残していなければ、プールは強い課題より損である。理由がノイズでも課題の違いでも、この文は式から出る。</p>
<p><strong>低い \(r\) には2つの読みがある。</strong> 測り損ないだけなら、誤差は \(\rho\) を0へ近づけるが符号は変えない。課題が別のものを引き出しているなら、符号は逆になりうる。\(r\) だけでは分けられない。分けられるのは符号である。</p>
<p><strong>符号が逆になる組がある。</strong> 21対のうち8対。例: 長いやり取り鎖と SA は、共同注意で \(+0.34\)、自由遊びで \(-0.25\)、\(r\approx 0\)、プール \(\rho_z\approx 0.05\)。測り損ないだけではこの向きの逆は出ない。存在の確認であり、有病率ではない。</p>
<p><strong>平均しても区間は狭まらない。点だけが0へ動く。</strong> 強い課題の区間が0を含まないのは 7/10。プールの区間は 10/10 が0を跨ぐ。幅の中央はほぼ同じである。</p>
<p><strong>反転は検査者開始の長いやり取りの側にある。</strong> 共同注意の検査者開始は SA \(+0.36\)（0を含まない）。自由遊びの検査者開始は \(-0.37\)（0を含まない）。子ども開始はどちらも0を含む。共同注意で検査者開始の長さ4以上の鎖が1本でもあった 15/55人は、SA 中央 8対6、CSS 中央 6対3。</p>
<p><strong>区間の長さだけでは符号は消えない。</strong> \(r\) と短いほうの課題長の関係はほぼ平坦（Spearman 0.06）。反転8対×2側は、長さを除いたあと 16/16 で符号が残る。</p>
<p>全セルの \(\lambda \lt \lambda^*\) が 81% という集計は使わない。重症度の名札を入れ替えても中央 82% になる。</p>
</div>
"""

body4 = rf"""
<h1>実験3 · 同じ名前の特徴を課題のあいだで平均すると何が起きるか</h1>
<p class="sub">59人 · 仮定・操作・結果。導出の正本: <a href="../hetero/実験3_数式と手続き.md">実験3_数式と手続き.md</a></p>
{EXP3_PROSE}
<div class="stats">
  <div class="stat"><b>{r_all_med:.2f}</b><span>全 {n_r_valid} 対の r の中央（重症度なし）</span></div>
  <div class="stat"><b>{lam_q25:.2f}–{lam_q75:.2f}</b><span>その r に対応する λ*（弱い側が残すべき割合）</span></div>
  <div class="stat"><b>16/16</b><span>反転8対×2側、区間長を除いても符号が残る</span></div>
  <div class="stat"><b>8 / 13</b><span>21対のうち符号が逆 / 同じ</span></div>
</div>
<p><img class="fig" src="fig_exp3_main.png" alt="(a) 全課題対の r の分布 (b) 21点が (λ,r) のどこに落ちたか (c) プールで点だけが0へ"/></p>
<p class="note">(a) 同名特徴が課題間でどれだけ同じ並びか。中央 0.25。(b) 21対は実験1の当たりからの例示。(c) 平均すると点だけが 0 へ動き、区間の幅はほぼ同じ。</p>
<h2 class="sec">符号が逆だった組み合わせ</h2>
<p class="note">21対のうち8対が逆。残り2行は21対の切り方には入っていない。そわそわ×CSS は同じ向きでも \(\lambda={fid_l:.3f} \lt \lambda^*={fid_ls:.3f}\) で損。</p>
<div class="table-wrap"><table><thead><tr>
<th>現象</th><th>特徴</th><th>得点</th><th>正側</th><th>負側</th><th>n<sub>∩</sub></th><th>ρ 正</th><th>ρ 負</th><th>r</th>
</tr></thead><tbody id="tab1Body"></tbody></table></div>
<h2 class="sec">開始者（検査者が切り出した鎖 / 子どもが切り出した鎖）</h2>
<p class="note">共同注意で検査者開始の長さ4以上の鎖が1本以上あったのは 15/55。SA 中央 8対6、CSS 中央 6対3。子ども開始の区間は0を含む。</p>
<p><img class="fig" src="fig_initiator.png" alt="鎖の開始者分解（CI付き）"/></p>
<div class="table-wrap"><table><thead><tr>
<th>課題</th><th>得点</th><th>開始者</th><th>n</th><th>ρ [CI]</th><th>ρ | 区間長</th>
</tr></thead><tbody id="initBody"></tbody></table></div>
<div class="table-wrap"><table><thead><tr>
<th>課題</th><th>開始者</th><th>セッション</th><th>鎖の中央</th><th>≥4 鎖の中央</th><th>≥4 が 0 の割合</th>
</tr></thead><tbody id="initNBody"></tbody></table></div>
<h2 class="sec">課題の長さ</h2>
<p class="note">\(r\) 対 短いほうの課題長: Spearman {rvd_rho:.2f}。四分位の \(r\) 中央は {rvd_bin_txt}。反転8対×2側は長さを除いたあと 16/16 で符号が残る。</p>
<div class="table-wrap"><table><thead><tr>
<th>特徴</th><th>得点</th><th>側</th><th>課題</th><th>ρ</th><th>ρ | duration</th><th>レート?</th>
</tr></thead><tbody id="partBody"></tbody></table></div>
<p><img class="fig" src="fig_r_vs_dur.png" alt="r 対 区間長"/></p>
<p><img class="fig" src="fig_duration_y.png" alt="課題長と重症度の相関"/></p>
<div class="table-wrap"><table><thead><tr>
<th>課題</th><th>n</th><th>SA</th><th>RRB</th><th>CSS</th>
</tr></thead><tbody id="durYBody"></tbody></table></div>
<div class="table-wrap"><table><thead><tr>
<th>課題</th><th>n</th><th>Q1</th><th>中央 (s)</th><th>Q3</th><th>min</th><th>max</th>
</tr></thead><tbody id="durBody"></tbody></table></div>
<h2 class="sec">共同注意の欠測</h2>
<p class="note">鎖の解析は51人。未実施4 + 短すぎて非有限4。8人とも ADOS 得点はある。欠測 vs 51人: SA 中央 5 vs 7、CSS 2 vs 4。欠測のほうが重い、という偏りは見えない。</p>
<h2 class="sec">使わない集計</h2>
<p class="note">向き付けたセルの \(\lambda \lt \lambda^*\) は観測 {frac_worse_u:.0%}、名札入れ替えの中央 {lam_null:.0%}。区別できないので使わない。</p>
<p><img class="fig" src="fig_lambda_perm.png" alt="λ&lt;λ* 割合の y シャッフル帰無"/></p>
<p><img class="fig" src="fig_q_ranks.png" alt="201族の Q の p 順位"/></p>
<div class="table-wrap"><table><thead><tr>
<th>特徴</th><th>得点</th><th>現象</th><th>p</th><th>順位 / 201</th>
</tr></thead><tbody id="qBody"></tbody></table></div>
<h2 class="sec">21対（実験1の当たりから切った記述）</h2>
<p class="note">\(\Delta\rho\) と区間。検定しない。</p>
<div class="filters">
  <select id="fTgt"></select>
  <select id="fSrc"></select>
  <select id="fFlip"></select>
  <span class="count" id="fCount"></span>
</div>
<div class="table-wrap"><table><thead><tr>
<th>特徴</th><th>得点</th><th>情報源</th><th>課題 A</th><th>課題 B</th><th>n<sub>∩</sub></th><th>ρ<sub>A</sub></th><th>ρ<sub>B</sub></th><th>r</th><th>λ</th><th>Δρ [CI]</th>
</tr></thead><tbody id="allBody"></tbody></table></div>
"""

js4 = f"""
const PAIRS = {dumps(rows21)};
const FLIPS = {dumps(rows10)};
const VERIFY = {dumps(rows_v)};
const QR = {dumps(rows_q)};
const DUR = {dumps(rows_dur)};
const DURY = {dumps(rows_dur_y)};
const PART = {dumps(rows_part)};
const INIT = {dumps(rows_init)};
const INITN = {dumps(rows_init_n)};
document.getElementById("tab1Body").innerHTML = FLIPS.map(h=>
  `<tr><td>${{esc(h.phen)}}</td><td>${{esc(h.featureJa)}} <code>${{esc(h.feature)}}</code></td><td>${{esc(h.target)}}</td><td>${{esc(h.posJa)}}</td><td>${{esc(h.negJa)}}</td><td class="num">${{h.n}}</td><td class="num">${{fmtRho(h.rhoPos)}}</td><td class="num">${{fmtRho(h.rhoNeg)}}</td><td class="num">${{h.r==null?"—":Number(h.r).toFixed(2)}}</td></tr>`
).join("");
document.getElementById("durBody").innerHTML = DUR.map(h=>
  `<tr class="${{[3,6,7,10].includes(h.task)?'hit':''}}"><td>${{esc(h.ja)}}</td><td class="num">${{h.n}}</td><td class="num">${{h.q25==null?"—":h.q25}}</td><td class="num">${{h.median}}</td><td class="num">${{h.q75==null?"—":h.q75}}</td><td class="num">${{h.min}}</td><td class="num">${{h.max}}</td></tr>`
).join("");
document.getElementById("durYBody").innerHTML = DURY.map(h=>
  `<tr class="${{[3,7].includes(h.task)?'hit':''}}"><td>${{esc(h.ja)}}</td><td class="num">${{h.n}}</td><td class="num">${{fmtRho(h.SA)}}</td><td class="num">${{fmtRho(h.RRB)}}</td><td class="num">${{fmtRho(h.CSS)}}</td></tr>`
).join("");
document.getElementById("partBody").innerHTML = PART.map(h=>
  `<tr class="${{h.ok?'pos':'neg'}}"><td>${{esc(h.featureJa)}}</td><td>${{esc(h.target)}}</td><td>${{esc(h.side)}}</td><td>${{esc(h.ja)}}</td><td class="num">${{fmtRho(h.rho)}}</td><td class="num">${{fmtRho(h.rhoP)}}</td><td>${{h.rate?"rate":"no"}}</td></tr>`
).join("");
document.getElementById("initBody").innerHTML = INIT.map(h=>
  `<tr><td>${{esc(h.taskJa)}}</td><td>${{esc(h.target)}}</td><td>${{esc(h.who)}}</td><td class="num">${{h.n}}</td><td class="num">${{fmtRho(h.rho)}} <span class="muted">${{fmtCI(h.lo,h.hi)}}</span></td><td class="num">${{fmtRho(h.rhoP)}}</td></tr>`
).join("");
document.getElementById("initNBody").innerHTML = INITN.map(h=>
  `<tr><td>${{esc(h.taskJa)}}</td><td>${{esc(h.who)}}</td><td class="num">${{h.n}}</td><td class="num">${{h.medN}}</td><td class="num">${{h.medGe4}}</td><td class="num">${{h.zero==null?"—":h.zero+"%"}}</td></tr>`
).join("");
document.getElementById("qBody").innerHTML = QR.map(h=>
  `<tr><td><code>${{esc(h.feature)}}</code></td><td>${{esc(h.target)}}</td><td>${{esc(h.phen)}}</td><td class="num">${{fmtP(h.p)}}</td><td class="num">${{h.rank}} / ${{h.nFam}}</td></tr>`
).join("");
const fTgt=document.getElementById("fTgt"), fSrc=document.getElementById("fSrc"), fFlip=document.getElementById("fFlip");
fTgt.innerHTML = `<option value="all">得点: すべて</option><option value="SA">SA</option><option value="RRB">RRB</option><option value="CSS">CSS</option>`;
fSrc.innerHTML = `<option value="all">情報源: すべて</option><option value="child">子ども</option><option value="examiner">検査者</option><option value="dyad">二人</option>`;
fFlip.innerHTML = `<option value="all">符号: すべて</option><option value="flip">反転のみ</option><option value="same">同符号のみ</option>`;
function filtered() {{
  return PAIRS.filter(h =>
    (fTgt.value==="all"||h.target===fTgt.value) &&
    (fSrc.value==="all"||h.source===fSrc.value) &&
    (fFlip.value==="all"|| (fFlip.value==="flip"?h.flip:!h.flip))
  );
}}
function render() {{
  const rows = filtered();
  document.getElementById("fCount").textContent = rows.length + " / " + PAIRS.length + " 件";
  document.getElementById("allBody").innerHTML = rows.map(h=>`<tr class="${{h.poolWorse?'neg':'pos'}}"><td>${{esc(h.featureJa)}}</td><td>${{esc(h.target)}}</td><td>${{esc(SRC_JA[h.source]||h.source)}}</td><td>${{esc(h.taskAJa)}}</td><td>${{esc(h.taskBJa)}}</td><td class="num">${{h.n}}</td><td class="num">${{fmtRho(h.rhoA)}}</td><td class="num">${{fmtRho(h.rhoB)}}</td><td class="num">${{h.r==null?"—":Number(h.r).toFixed(2)}}</td><td class="num">${{h.lambda==null?"—":Number(h.lambda).toFixed(2)}}</td><td class="num">${{fmtRho(h.dz)}} <span class="muted">${{fmtCI(h.dzLo,h.dzHi)}}</span></td></tr>`).join("");
}}
[fTgt,fSrc,fFlip].forEach(el=>el.addEventListener("change", render));
render();
"""

(OUT / "exp3.html").write_text(page("実験3 · 課題を跨ぐ平均", "exp3", body4, js4), encoding="utf-8")

# ---------- index ----------
body_i = f"""
<h1>ADOS 実験結果（手動課題分割）</h1>
<p class="sub">n=59 · data/task_segments.json · 本線は実験1（名簿）· 実験3（対比）· 実験2（LOPO 校正）。Ridge は補助。</p>
<div class="callout">論文用の手続き・データ・限界の正本は
<a href="../論文用_確認事項_実験1と実験2.md">../論文用_確認事項_実験1と実験2.md</a>。
実験3の表の正本は <a href="../hetero/README.md">../hetero/README.md</a>。
数式は <a href="../hetero/実験3_数式と手続き.md">../hetero/実験3_数式と手続き.md</a>。</div>
<div class="grid2">
  <div class="card"><h2><a href="exp1.html">実験1 · 単変量スクリーニング</a></h2>
  <p>FDR 当たり {meta1['n_main_fdr_hits']}（正 {int((hits.rho>0).sum())} / 負 {int((hits.rho<0).sum())}）。探索的な名簿。課題・得点・情報源でフィルタ。</p></div>
  <div class="card"><h2><a href="exp3.html">実験3 · 課題を跨ぐ平均</a></h2>
  <p>同名特徴の課題間の一致（中央 0.25）。平均すると重症度との関係はどうなるか。符号が逆になる組がある。</p></div>
  <div class="card"><h2><a href="exp2.html">実験2 · LOPO（校正）</a></h2>
  <p>41セル、FDR 通過 <b>2</b>。学習側で1本選び sign(ρ)×x。点推定の校正。独立標本ではない。</p></div>
  <div class="card"><h2><a href="exp2_ridge.html">実験2補助 · Ridge</a></h2>
  <p>実験1の当たりをそのまま Ridge。41セル、FDR 通過 37。主結果にはしない。</p></div>
</div>
"""
(OUT / "index.html").write_text(page("実験結果共有 · 手動課題分割", "index", body_i, ""), encoding="utf-8")

print("wrote", sorted(p.name for p in OUT.iterdir()))
