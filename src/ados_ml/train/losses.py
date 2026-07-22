"""Multi-task losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_l1(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=beta)


def ordinal_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cumulative ordinal loss.

    logits: (B, K-1) for thresholds y >= 1 .. y >= K-1
    y: (B,) integer levels in 0..K-1
    """
    # targets[b, r] = 1 if y[b] >= r+1
    k_minus_1 = logits.shape[-1]
    thresholds = torch.arange(1, k_minus_1 + 1, device=logits.device, dtype=y.dtype)
    target = (y.unsqueeze(-1) >= thresholds).float()
    return F.binary_cross_entropy_with_logits(logits, target)


def bce_with_logits(
    logit: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor | None = None
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)


def decode_ordinal(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (integer_pred, expected_value)."""
    probs = torch.sigmoid(logits)
    expected = probs.sum(dim=-1)
    integer = (probs > 0.5).sum(dim=-1)
    return integer.float(), expected
