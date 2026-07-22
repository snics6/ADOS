"""Train / evaluate one multi-task model on a fold."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ados_ml.eval.metrics import binary_metrics, ordinal_metrics, regression_metrics
from ados_ml.models.multitask import MultiTaskADOSModel
from ados_ml.train.losses import bce_with_logits, decode_ordinal, ordinal_bce, smooth_l1
from ados_ml.train.preprocess import FeaturePreprocessor, LabelZScorer


def _to_loader(
    X: np.ndarray,
    y: dict[str, np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors = [torch.from_numpy(X.astype(np.float32))]
    for key in ("SA", "RRB", "C2", "B1", "B12"):
        tensors.append(torch.from_numpy(y[key].astype(np.float32)))
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def compute_loss(
    out: dict[str, torch.Tensor],
    batch_y: dict[str, torch.Tensor],
    *,
    sa_z: torch.Tensor,
    rrb_z: torch.Tensor,
    loss_weights: dict[str, float],
    huber_beta: float,
    pos_weight_b1: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    parts = {}
    parts["SA"] = smooth_l1(out["SA"], sa_z, beta=huber_beta)
    parts["RRB"] = smooth_l1(out["RRB"], rrb_z, beta=huber_beta)
    parts["C2"] = ordinal_bce(out["C2"], batch_y["C2"])
    parts["B1"] = bce_with_logits(out["B1"], batch_y["B1"], pos_weight=pos_weight_b1)
    parts["B12"] = ordinal_bce(out["B12"], batch_y["B12"])
    total = sum(loss_weights[k] * parts[k] for k in parts)
    return total, {k: float(v.detach().cpu()) for k, v in parts.items()}


@torch.no_grad()
def predict_numpy(
    model: MultiTaskADOSModel,
    X: np.ndarray,
    *,
    device: torch.device,
    task_mask: torch.Tensor | np.ndarray | None = None,
    sa_scaler: LabelZScorer,
    rrb_scaler: LabelZScorer,
) -> dict[str, Any]:
    model.eval()
    x = torch.from_numpy(X.astype(np.float32)).to(device)
    mask = None
    if task_mask is not None:
        if isinstance(task_mask, np.ndarray):
            mask = torch.from_numpy(task_mask.astype(bool))
        else:
            mask = task_mask
        mask = mask.to(device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(x.size(0), -1)
    out = model(x, task_mask=mask)
    sa = sa_scaler.inverse(out["SA"].cpu().numpy())
    rrb = rrb_scaler.inverse(out["RRB"].cpu().numpy())
    c2_int, c2_exp = decode_ordinal(out["C2"])
    b12_int, b12_exp = decode_ordinal(out["B12"])
    b1_prob = torch.sigmoid(out["B1"]).cpu().numpy()
    b1_pred = (b1_prob >= 0.5).astype(np.float64)
    attn = {k: v.cpu().numpy() for k, v in out["attention"].items()}
    return {
        "SA": sa,
        "RRB": rrb,
        "C2_int": c2_int.cpu().numpy(),
        "C2_exp": c2_exp.cpu().numpy(),
        "B12_int": b12_int.cpu().numpy(),
        "B12_exp": b12_exp.cpu().numpy(),
        "B1_prob": b1_prob,
        "B1_pred": b1_pred,
        "attention": attn,
    }


def evaluate_predictions(y: dict[str, np.ndarray], pred: dict[str, Any]) -> dict[str, Any]:
    return {
        "SA": regression_metrics(y["SA"], pred["SA"]),
        "RRB": regression_metrics(y["RRB"], pred["RRB"]),
        "C2": ordinal_metrics(y["C2"], pred["C2_int"], pred["C2_exp"]),
        "B12": ordinal_metrics(y["B12"], pred["B12_int"], pred["B12_exp"]),
        "B1": binary_metrics(y["B1"], pred["B1_prob"], pred["B1_pred"]),
    }


def early_stop_score(metrics: dict[str, Any]) -> float:
    """Higher is better."""
    return float(
        -(
            metrics["SA"]["mae"]
            + metrics["RRB"]["mae"]
            + metrics["C2"]["mae"]
            + metrics["B12"]["mae"]
        )
        / 4.0
        + metrics["B1"]["balanced_accuracy"]
    )


def train_one_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: dict[str, np.ndarray],
    y_val: dict[str, np.ndarray],
    *,
    cfg: dict,
    device: torch.device,
    task_mask: np.ndarray | None = None,
) -> tuple[MultiTaskADOSModel, FeaturePreprocessor, LabelZScorer, LabelZScorer, dict]:
    pre = FeaturePreprocessor().fit(X_train)
    Xtr = pre.transform(X_train)
    Xva = pre.transform(X_val)

    sa_sc = LabelZScorer().fit(y_train["SA"])
    rrb_sc = LabelZScorer().fit(y_train["RRB"])

    n_pos = max(float(np.sum(y_train["B1"] == 1)), 1.0)
    n_neg = max(float(np.sum(y_train["B1"] == 0)), 1.0)
    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)

    model_cfg = cfg.get("model", {})
    model = MultiTaskADOSModel(
        in_dim=Xtr.shape[-1],
        n_tasks=Xtr.shape[1],
        hidden=int(model_cfg.get("encoder_hidden", 64)),
        encoder_layers=int(model_cfg.get("encoder_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        shared_trunk=bool(model_cfg.get("shared_trunk", False)),
    ).to(device)

    train_cfg = cfg.get("train", {})
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    loss_w = cfg.get("loss_weights", {})
    lw = {
        "SA": float(loss_w.get("SA", 1.0)),
        "RRB": float(loss_w.get("RRB", 0.5)),
        "C2": float(loss_w.get("C2", 1.0)),
        "B1": float(loss_w.get("B1", 0.5)),
        "B12": float(loss_w.get("B12", 0.75)),
    }
    huber_beta = float(train_cfg.get("huber_beta", 1.0))
    batch_size = int(train_cfg.get("batch_size", 8))
    max_epochs = int(train_cfg.get("max_epochs", 100))
    patience = int(train_cfg.get("early_stopping_patience", 20))

    loader = _to_loader(Xtr, y_train, batch_size=batch_size, shuffle=True)

    mask_t = None
    if task_mask is not None:
        mask_t = torch.from_numpy(task_mask.astype(bool))

    best_state = None
    best_score = -1e18
    best_metrics: dict = {}
    bad = 0

    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            xb = batch[0].to(device)
            yb = {
                "SA": batch[1].to(device),
                "RRB": batch[2].to(device),
                "C2": batch[3].to(device),
                "B1": batch[4].to(device),
                "B12": batch[5].to(device),
            }
            sa_z = (yb["SA"] - sa_sc.mean) / sa_sc.std
            rrb_z = (yb["RRB"] - rrb_sc.mean) / rrb_sc.std

            bm = None
            if mask_t is not None:
                bm = mask_t.to(device).unsqueeze(0).expand(xb.size(0), -1)
            out = model(xb, task_mask=bm)
            loss, _ = compute_loss(
                out,
                yb,
                sa_z=sa_z,
                rrb_z=rrb_z,
                loss_weights=lw,
                huber_beta=huber_beta,
                pos_weight_b1=pos_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        pred = predict_numpy(
            model,
            Xva,
            device=device,
            task_mask=mask_t,
            sa_scaler=sa_sc,
            rrb_scaler=rrb_sc,
        )
        metrics = evaluate_predictions(y_val, pred)
        score = early_stop_score(metrics)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pre, sa_sc, rrb_sc, {
        "best_score": best_score,
        "val_metrics": best_metrics,
        "epochs_trained": epoch + 1,
    }
