# ADOS-exam

ADOS-2 Module 2 録画からの自動評価・分析リポジトリ。

## 現行（第4世代）

個別特徴の発見・回帰による確認・課題差（実験1・2・3・4）。

手順: 特徴を一度出す → 実験1・2・3・4はその表を読む。

```
./venv/bin/python -u scripts/extract_task_features.py
./venv/bin/python scripts/run_exp1_univariate.py
./venv/bin/python -u scripts/run_exp1_target_overlap.py
./venv/bin/python -u scripts/run_exp2_ridge.py
./venv/bin/python -u scripts/run_exp3_transplant.py --labels-only
./venv/bin/python -u scripts/run_exp3_transplant.py
./venv/bin/python -u scripts/run_exp3_resolution.py
./venv/bin/python -u scripts/run_exp4_pool.py
./venv/bin/python -u scripts/run_group_check.py
./venv/bin/python -u scripts/run_drop_book_story.py
```

| 何 | 入口 | 書き先 |
|---|---|---|
| 特徴（動き・発話の時間構造） | `src/ados_extract/dynamics/` | `outputs/features/dynamics/features_task.csv` |
| 特徴（時間窓・返事・身振り・テキスト） | `src/ados_extract/windows.py` | `outputs/features/windows/features_task.csv` |
| 実験1 | `scripts/run_exp1_univariate.py` | `outputs/exp1/` |
| 点数のまたぎ | `scripts/run_exp1_target_overlap.py` | `outputs/exp1/target_overlap_*.csv` |
| 実験2 | `scripts/run_exp2_ridge.py` | `outputs/exp2/` |
| 実験3 | `scripts/run_exp3_transplant.py` | `outputs/exp3/` |
| 実験3の解像度 | `scripts/run_exp3_resolution.py` | `outputs/exp3_resolution/` |
| 実験4 | `scripts/run_exp4_pool.py` | `outputs/exp4/` |
| 群の確認 | `scripts/run_group_check.py` | `outputs/group_check/` |
| 本のストーリーを除く | `scripts/run_drop_book_story.py` | `outputs/drop_book_story/` |

- 手順書: [`docs/実験手順書_個別特徴と課題差.md`](docs/実験手順書_個別特徴と課題差.md)
- 過去知見の要約: [`docs/これまでの知見まとめ.md`](docs/これまでの知見まとめ.md)
- コホート: `configs/cohorts/features_63.yaml`（n=63）

2026-08-18 時点: 抽出・実験1〜4・点数のまたぎ・解像度・群の確認・本のストーリーを除く確認は済．列平均の旧実験3は [`old/src/ados_ffm/exp3_nearfar.py`](old/src/ados_ffm/exp3_nearfar.py)．

## アーカイブ

| 世代 | 場所 |
|---|---|
| 第0世代（〜2026-07） | `old_files/00_pre_direction1/` |
| 第1世代（マトリクス / Phase） | `old_files/01_direction1_v1_v2/` |
| 第2世代（セグメント・動態・安定性） | `old_files/02_ados_seg_202608/` |
| 第3世代・旧実験2/3・整理前の第4世代出力 | [`old/`](old/README.md) |

索引: [`old_files/README.md`](old_files/README.md)

## その他

- 生データ: `data/`
- 参考文献: `references/`
- 詳細な旧ドキュメント: `docs/old/`
