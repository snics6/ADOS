---
description: 変更・新規のPythonファイルだけに ruff check / format をかける
---
`git status --short` で変更・新規のファイル一覧を確認し，追跡・未追跡を問わず対象になっている `.py` ファイルだけを対象に `./venv/bin/ruff check <files>` と `./venv/bin/ruff format <files>` を実行して。プロジェクト全体を一括で整形しない（既存コード全体はまだ ruff を通していないため，触っていない箇所は変えない）。
