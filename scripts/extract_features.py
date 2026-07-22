#!/usr/bin/env python3
"""Extract per-task child/dyad features for the configured cohort."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ados_ml.data.cohort import load_cohort
from ados_ml.features.extract import extract_cohort
from ados_ml.utils.config import load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Experiment config YAML",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help="Override cohort YAML (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: outputs/features/<timestamp>/features.csv)",
    )
    parser.add_argument(
        "--embed-text",
        action="store_true",
        help="Compute frozen speech embeddings (requires sentence-transformers)",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for text encoder (cpu|cuda)",
    )
    parser.add_argument("--tau-pose", type=float, default=0.25)
    parser.add_argument("--tau-face", type=float, default=0.25)
    parser.add_argument("--tau-sp", type=float, default=0.3)
    args = parser.parse_args()

    root = project_root()
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    cfg = load_yaml(cfg_path)

    cohort_path = args.cohort
    if cohort_path is None:
        cohort_path = Path(cfg["paths"]["cohort"])
    if not cohort_path.is_absolute():
        cohort_path = root / cohort_path
    cohort = load_cohort(cohort_path)

    data_root = Path(cfg["paths"]["data_root"])
    if not data_root.is_absolute():
        data_root = root / data_root

    embed_fn = None
    embed_meta = {"enabled": False}
    if args.embed_text:
        from ados_ml.features.text_encoder import TextEncoder

        encoder = TextEncoder(args.text_model, device=args.device)
        embed_fn = encoder
        embed_meta = {
            "enabled": True,
            "model": args.text_model,
            "device": args.device,
            "dim": encoder.dim,
        }

    df = extract_cohort(
        cohort["participant_ids"],
        data_root,
        tau_pose=args.tau_pose,
        tau_face=args.tau_face,
        tau_sp=args.tau_sp,
        embed_fn=embed_fn,
    )

    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = root / "outputs" / "features" / stamp
        out_path = out_dir / "features.csv"
    else:
        out_path = args.output if args.output.is_absolute() else root / args.output
        out_dir = out_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep leading zeros in participant_id
    df.to_csv(out_path, index=False, quoting=1)  # csv.QUOTE_ALL

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg_path),
        "cohort": str(cohort_path),
        "cohort_name": cohort.get("name"),
        "n_participants": len(cohort["participant_ids"]),
        "n_rows": int(len(df)),
        "tau_pose": args.tau_pose,
        "tau_face": args.tau_face,
        "tau_sp": args.tau_sp,
        "text_embedding": embed_meta,
        "feature_doc": "docs/features.md",
        "output": str(out_path),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path}  shape={df.shape}")
    print(f"Meta {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
