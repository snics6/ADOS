# ADOS-exam

ADOS-2 Module 2 録画からの自動評価・分析リポジトリ。

## 現行

個別特徴の発見（実験1）、1人抜きの確認（実験2）、同じ名前の特徴を課題間で比べる（実験3）。実験2の Ridge は補助。旧い移植・プールは `old/20260901_unused_transplant_pool/`。

手順: 特徴を一度出す（`data/task_segments.json` の手動課題区間のみ）→ 実験1・2・3はその表を読む。

```
./venv/bin/python -u scripts/extract_task_features.py
./venv/bin/python scripts/run_exp1_univariate.py
./venv/bin/python -u scripts/run_exp1_target_overlap.py
./venv/bin/python -u scripts/run_exp2.py
./venv/bin/python -u scripts/run_hetero.py
./venv/bin/python -u scripts/run_exp3_findings.py
# 補助（実験1の当たりをそのまま Ridge）:
# ./venv/bin/python -u scripts/run_exp2_ridge.py
```

| 何 | 入口 | 書き先 |
|---|---|---|
| 特徴（動き・発話の時間構造） | `src/ados_extract/dynamics/` | `outputs/features/dynamics/features_task.csv` |
| 特徴（時間窓・返事・身振り・テキスト） | `src/ados_extract/windows.py` | `outputs/features/windows/features_task.csv` |
| 実験1 | `scripts/run_exp1_univariate.py` | `outputs/exp1/` |
| 点数のまたぎ | `scripts/run_exp1_target_overlap.py` | `outputs/exp1/target_overlap_*.csv` |
| **実験2（LOPO）** | `scripts/run_exp2.py` | `outputs/exp2/` |
| 実験2の補助（実験1の当たりをそのまま Ridge） | `scripts/run_exp2_ridge.py` | `outputs/exp2_ridge/` |
| **実験3（\(r\)、\(\lambda^*\)、符号反転の例示）** | `scripts/run_hetero.py` + `scripts/run_exp3_findings.py` | `outputs/hetero/` |

- 手順書: [`docs/実験手順書_個別特徴と課題差.md`](docs/実験手順書_個別特徴と課題差.md)
- 論文用の確認事項: [`outputs/論文用_確認事項_実験1と実験2.md`](outputs/論文用_確認事項_実験1と実験2.md)
- 実験3の結果: [`outputs/hetero/README.md`](outputs/hetero/README.md)
- 実験3の数式: [`outputs/hetero/実験3_数式と手続き.md`](outputs/hetero/実験3_数式と手続き.md)
- 共有 HTML: [`outputs/sharing/index.html`](outputs/sharing/index.html)
- 過去知見の要約: [`docs/これまでの知見まとめ.md`](docs/これまでの知見まとめ.md)
- コホート: `configs/cohorts/features_63.yaml`（**n=59**）。課題区間正本: `data/task_segments.json`
- 2026-08-28 以前の実験出力は `outputs/old/20260828_pre_manual_segments/`（自動・旧分割ベース）

## アーカイブ

| 世代 | 場所 |
|---|---|
| 第0世代（〜2026-07） | `old_files/00_pre_direction1/` |
| 第1世代（マトリクス / Phase） | `old_files/01_direction1_v1_v2/` |
| 第2世代（セグメント・動態・安定性） | `old_files/02_ados_seg_202608/` |
| 第3世代・旧実験2/3・整理前の第4世代出力 | [`old/`](old/README.md) |
| 2026-08-31 に本線から外したもの | [`old/20260831_repo_cleanup/`](old/20260831_repo_cleanup/README.md) |
| 2026-09-01 に本線から外した移植・プール | [`old/20260901_unused_transplant_pool/`](old/20260901_unused_transplant_pool/README.md) |

索引: [`old_files/README.md`](old_files/README.md)

## その他

- 生データ: `data/`
- 参考文献: `references/`
- 詳細な旧ドキュメント: `docs/old/`
