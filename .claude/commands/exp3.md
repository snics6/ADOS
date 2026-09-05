---
description: 実験3（同名特徴の課題間比較）を run_hetero.py → run_exp3_findings.py の2段階で実行する
argument-hint: [--smoke]
---
以下を順に実行し，それぞれの出力を要約して報告して。

1. `./venv/bin/python -u scripts/run_hetero.py $ARGUMENTS`
2. `./venv/bin/python -u scripts/run_exp3_findings.py`（`run_exp3_findings.py` は `--smoke` 等の引数を受け付けないため，`$ARGUMENTS` はこちらには渡さない）

時間のかかる実行になりそうなら，事前に一言断ってから実行して。
