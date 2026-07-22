#!/usr/bin/env python3
"""Train multi-task models under participant-level 4-fold CV."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ados_ml.data.cohort import load_cohort
from ados_ml.train.experiment import run_experiment
from ados_ml.utils.config import load_yaml, project_root


def _latest_features(root: Path) -> Path:
    cand = sorted((root / "outputs" / "features").glob("*/features.csv"))
    if not cand:
        raise FileNotFoundError(
            "No features CSV found under outputs/features/. "
            "Run scripts/extract_features.py first."
        )
    return cand[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Feature CSV (default: latest under outputs/features/)",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda|cpu override")
    args = parser.parse_args()

    root = project_root()
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    cfg = load_yaml(cfg_path)

    cohort_path = root / cfg["paths"]["cohort"]
    cohort = load_cohort(cohort_path)

    features = args.features
    if features is None:
        features = _latest_features(root)
    elif not features.is_absolute():
        features = root / features

    labels_xlsx = root / cfg["paths"]["labels_xlsx"]
    labels_sheet = cfg["paths"]["labels_sheet"]

    stamp = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / cfg["paths"]["output_root"] / "runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config used for this run
    import shutil

    shutil.copy2(cfg_path, out_dir / "config.yaml")

    summary = run_experiment(
        features_csv=features,
        labels_xlsx=labels_xlsx,
        labels_sheet=labels_sheet,
        participant_ids=cohort["participant_ids"],
        cfg=cfg,
        output_dir=out_dir,
        device=args.device,
    )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg_path),
        "features": str(features),
        "output_dir": str(out_dir),
    }
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(summary.get("neural_full", {}), indent=2))
    print("explain:", json.dumps(summary.get("explain", {}), indent=2))
    if "sklearn_baseline" in summary:
        print("baseline:", json.dumps(summary["sklearn_baseline"], indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
