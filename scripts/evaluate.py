#!/usr/bin/env python3
"""Print a short summary for an existing run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
