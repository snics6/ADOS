"""Full 4-fold experiment with ablations and baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ados_ml.data.dataset import FeatureBundle, apply_ablation, build_feature_bundle
from ados_ml.data.labels import load_labels
from ados_ml.eval.baselines import run_sklearn_baselines
from ados_ml.eval.explain import c2_ablation_pass, c2_pretend_attention_pass
from ados_ml.train.cv import make_folds, sa_tertiles
from ados_ml.train.loop import evaluate_predictions, predict_numpy, train_one_fold
from ados_ml.utils.ids import normalize_participant_id


def _slice_y(y: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in y.items()}


def _mean_std(dicts: list[dict], path: tuple[str, ...]) -> dict[str, float]:
    vals = []
    for d in dicts:
        cur: Any = d
        for p in path:
            cur = cur[p]
        vals.append(float(cur))
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}


def run_experiment(
    *,
    features_csv: Path,
    labels_xlsx: Path,
    labels_sheet: str,
    participant_ids: list[str],
    cfg: dict,
    output_dir: Path,
    device: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    feats = pd.read_csv(features_csv, dtype={"participant_id": str})
    feats["participant_id"] = feats["participant_id"].map(normalize_participant_id)
    labels = load_labels(
        labels_xlsx, sheet=labels_sheet, participant_ids=participant_ids
    )
    bundle = build_feature_bundle(feats, labels)

    n_folds = int(cfg.get("cv", {}).get("n_folds", 4))
    strata = sa_tertiles(bundle.y["SA"])
    folds = make_folds(len(bundle.participant_ids), n_folds=n_folds, stratify=strata, seed=seed)

    explain_cfg = cfg.get("explain", {})
    do_baseline = bool(explain_cfg.get("sklearn_baseline", True))
    do_pretend_ablation = bool(explain_cfg.get("c2_pretend_ablation", True))
    do_dyad_ablation = bool(explain_cfg.get("dyad_ablation", True))

    fold_rows: list[dict] = []
    attn_c2_folds: list[np.ndarray] = []
    metrics_full: list[dict] = []
    metrics_nopretend: list[dict] = []
    metrics_nodyad: list[dict] = []
    metrics_base: list[dict] = []

    task_mask_full = None
    task_mask_nopretend = np.ones(len(bundle.task_ids), dtype=bool)
    task_mask_nopretend[bundle.pretend_task_index] = False

    for fold_i, (tr, te) in enumerate(folds):
        Xtr, Xte = bundle.X[tr], bundle.X[te]
        ytr, yte = _slice_y(bundle.y, tr), _slice_y(bundle.y, te)

        model, pre, sa_sc, rrb_sc, info = train_one_fold(
            Xtr, Xte, ytr, yte, cfg=cfg, device=dev, task_mask=task_mask_full
        )
        Xte_p = pre.transform(Xte)
        pred = predict_numpy(
            model, Xte_p, device=dev, sa_scaler=sa_sc, rrb_scaler=rrb_sc
        )
        m_full = evaluate_predictions(yte, pred)
        metrics_full.append(m_full)
        attn_c2 = pred["attention"]["C2"].mean(axis=0)
        attn_c2_folds.append(attn_c2)

        # Save per-fold predictions
        fold_df = pd.DataFrame(
            {
                "fold": fold_i,
                "participant_id": [bundle.participant_ids[i] for i in te],
                "SA_true": yte["SA"],
                "SA_pred": pred["SA"],
                "RRB_true": yte["RRB"],
                "RRB_pred": pred["RRB"],
                "C2_true": yte["C2"],
                "C2_pred": pred["C2_int"],
                "B12_true": yte["B12"],
                "B12_pred": pred["B12_int"],
                "B1_true": yte["B1"],
                "B1_prob": pred["B1_prob"],
                "B1_pred": pred["B1_pred"],
            }
        )
        fold_df.to_csv(output_dir / f"fold{fold_i}_predictions.csv", index=False)

        # Pretend ablation: retrain
        if do_pretend_ablation:
            Xtr_a = apply_ablation(Xtr, bundle, drop_pretend=True)
            Xte_a = apply_ablation(Xte, bundle, drop_pretend=True)
            model_a, pre_a, sa_a, rrb_a, _ = train_one_fold(
                Xtr_a,
                Xte_a,
                ytr,
                yte,
                cfg=cfg,
                device=dev,
                task_mask=task_mask_nopretend,
            )
            pred_a = predict_numpy(
                model_a,
                pre_a.transform(Xte_a),
                device=dev,
                task_mask=task_mask_nopretend,
                sa_scaler=sa_a,
                rrb_scaler=rrb_a,
            )
            metrics_nopretend.append(evaluate_predictions(yte, pred_a))

        if do_dyad_ablation:
            Xtr_d = apply_ablation(Xtr, bundle, drop_dyad=True)
            Xte_d = apply_ablation(Xte, bundle, drop_dyad=True)
            model_d, pre_d, sa_d, rrb_d, _ = train_one_fold(
                Xtr_d, Xte_d, ytr, yte, cfg=cfg, device=dev
            )
            pred_d = predict_numpy(
                model_d,
                pre_d.transform(Xte_d),
                device=dev,
                sa_scaler=sa_d,
                rrb_scaler=rrb_d,
            )
            metrics_nodyad.append(evaluate_predictions(yte, pred_d))

        if do_baseline:
            metrics_base.append(
                run_sklearn_baselines(Xtr, Xte, ytr, yte)
            )

        fold_rows.append(
            {
                "fold": fold_i,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "epochs": info["epochs_trained"],
                "val_score": info["best_score"],
                "SA_mae": m_full["SA"]["mae"],
                "RRB_mae": m_full["RRB"]["mae"],
                "C2_mae": m_full["C2"]["mae"],
                "B12_mae": m_full["B12"]["mae"],
                "B1_bal_acc": m_full["B1"]["balanced_accuracy"],
                "C2_attn_pretend": float(attn_c2[bundle.pretend_task_index]),
            }
        )

    # Aggregate attention
    attn_mean = np.mean(np.stack(attn_c2_folds, axis=0), axis=0)
    attn_check = c2_pretend_attention_pass(
        attn_mean, pretend_index=bundle.pretend_task_index, task_ids=bundle.task_ids
    )

    summary: dict[str, Any] = {
        "n_participants": len(bundle.participant_ids),
        "n_features_per_task": len(bundle.feature_names),
        "feature_names": bundle.feature_names,
        "device": str(dev),
        "folds": fold_rows,
        "neural_full": {
            "SA": _mean_std(metrics_full, ("SA", "mae")),
            "SA_spearman": _mean_std(metrics_full, ("SA", "spearman")),
            "RRB": _mean_std(metrics_full, ("RRB", "mae")),
            "C2": _mean_std(metrics_full, ("C2", "mae")),
            "C2_spearman": _mean_std(metrics_full, ("C2", "spearman")),
            "B12": _mean_std(metrics_full, ("B12", "mae")),
            "B1_bal_acc": _mean_std(metrics_full, ("B1", "balanced_accuracy")),
            "B1_auc": _mean_std(metrics_full, ("B1", "auc")),
        },
        "explain": {
            "c2_attention": attn_check,
        },
    }

    if metrics_nopretend:
        mae_full = summary["neural_full"]["C2"]["mean"]
        mae_np = _mean_std(metrics_nopretend, ("C2", "mae"))["mean"]
        summary["explain"]["c2_pretend_ablation"] = c2_ablation_pass(mae_full, mae_np)
        summary["neural_no_pretend_C2_mae"] = _mean_std(metrics_nopretend, ("C2", "mae"))

    if metrics_nodyad:
        summary["neural_no_dyad"] = {
            "SA": _mean_std(metrics_nodyad, ("SA", "mae")),
            "B1_bal_acc": _mean_std(metrics_nodyad, ("B1", "balanced_accuracy")),
            "B12": _mean_std(metrics_nodyad, ("B12", "mae")),
        }

    if metrics_base:
        summary["sklearn_baseline"] = {
            "SA": _mean_std(metrics_base, ("SA", "mae")),
            "RRB": _mean_std(metrics_base, ("RRB", "mae")),
            "C2": _mean_std(metrics_base, ("C2", "mae")),
            "B12": _mean_std(metrics_base, ("B12", "mae")),
            "B1_bal_acc": _mean_std(metrics_base, ("B1", "balanced_accuracy")),
        }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    return summary
