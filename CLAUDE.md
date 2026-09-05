# プロジェクト概要
自閉症スクリーニング検査であるADOSビデオから抽出した特徴量から，ADOSスコアを予測するプロジェクト
（ADOS-2 Module 2。実験1=個別特徴の発見、実験2=1人抜き(LOPO)確認、実験3=同名特徴の課題間比較）

# よく使うコマンド
実行は必ず venv の Python を明示する（`./venv/bin/python ...`）。
- 依存インストール: `./venv/bin/pip install -r requirements.txt`
- 特徴量の抽出（まず1回，約20分）: `./venv/bin/python -u scripts/extract_task_features.py`
- 実験1（個別特徴の発見）: `./venv/bin/python scripts/run_exp1_univariate.py`
  - 課題またぎの点数確認: `./venv/bin/python scripts/run_exp1_target_overlap.py`
  - 上位特徴のプロット: `./venv/bin/python scripts/plot_exp1_hits.py`
- 実験2 (LOPO，本線): `./venv/bin/python -u scripts/run_exp2.py`（フル実行は数時間規模になりうる）
  - 実験2補助（Exp1の当たりをそのままRidge，本線ではない）: `./venv/bin/python -u scripts/run_exp2_ridge.py`
- 実験3（課題間の異質性）: `./venv/bin/python -u scripts/run_hetero.py` → `./venv/bin/python -u scripts/run_exp3_findings.py`
  - `run_hetero.py`は内部で`run_exp3_findings.py`と同じfindings計算を実行済みなので，単体再実行は主に再現性確認・findings単独更新用
  - 式3の検証: `./venv/bin/python -u scripts/run_pool_verify.py`
- 共有用HTML生成: `./venv/bin/python scripts/build_sharing_html.py`
- Lint: `./venv/bin/ruff check src scripts`
- 整形: `./venv/bin/ruff format src scripts`
- 動作確認: `--smoke`対応スクリプト（`run_exp1_univariate.py`, `run_exp2.py`, `run_exp2_ridge.py`, `run_hetero.py`, `run_pool_verify.py`）は少量データで最後まで通るか確認できる。例: `./venv/bin/python scripts/run_exp1_univariate.py --smoke`
  - `extract_task_features.py`と`run_exp3_findings.py`は`--smoke`非対応

# 環境
- Python 3.12.3、仮想環境はプロジェクト直下 `./venv/`（`./venv/bin/python` で実行）
- 依存は venv に pip で導入。宣言は `pyproject.toml` と `requirements.txt`
- パッケージは editable インストールしていない。各スクリプトが冒頭で `sys.path` に `src/` を足して `ados_extract` / `ados_ffm` を import する
- GPU あり（NVIDIA RTX A5000, 24GB）。torch 2.13.0 導入済み
- 主要ライブラリ: numpy / pandas / scikit-learn / scipy / matplotlib / seaborn / pyyaml / openpyxl / tqdm

# コーディング規約
- フォーマッタ・命名・import は既存ファイルに合わせる
- 整形と lint は `ruff`（`ruff format` / `ruff check`）を使う。現状まだ使っていないので、既存コード全体を一度に整形しない。新規・変更した箇所だけ整える
- 各モジュール・スクリプトは `from __future__ import annotations` で始める（モジュール docstring がある場合はその直後。`__init__.py` は対象外）。import は 標準ライブラリ → サードパーティ の順
- 実行入口は `scripts/`、再利用ロジックは `src/ados_extract/`（特徴抽出）・`src/ados_ffm/`（実験・モデル）に置く

# 注意・やってはいけないこと
- 学習など長時間処理は勝手に実行しない。まず実行内容を提示する。
- data/ は生データ。編集・削除しない。
- outputs/ は実験の生成物（抽出済み特徴CSVを含む）。手で書き換えない。再生成はスクリプト経由。
- old/ , old_files/ , docs/old/ は過去世代のアーカイブ。参照はしても変更しない。
- data/ , outputs/ , old/ , old_files/ , docs/old/ は .gitignore 済み。ここへ commit しようとしない。

# ディレクトリ（自明でないもの）
- src/ados_extract/: 特徴量の抽出（dynamics=動き・発話の時間構造、windows=時間窓・返事・身振り・テキスト）
- src/ados_ffm/: 実験本体（exp1_univariate, exp2, hetero, exp3_findings, ridge_cv, metrics）
- scripts/: 上記を呼ぶ実行入口
- configs/cohorts/features_63.yaml: 対象コホート定義（n=59）
- data/: 被験者ごとの数値IDフォルダ（生データ）。課題区間の正本は data/task_segments.json
- outputs/: 実験出力。features/ に抽出済み特徴CSV
- references/: 参考文献・メモ
- docs/: 手順書・知見まとめ
