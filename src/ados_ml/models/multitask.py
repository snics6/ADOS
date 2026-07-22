"""Multi-task model with per-target task attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


TARGET_NAMES = ("SA", "RRB", "C2", "B1", "B12")


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for i in range(max(n_layers, 1)):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskADOSModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        *,
        n_tasks: int = 4,
        hidden: int = 64,
        encoder_layers: int = 2,
        dropout: float = 0.15,
        shared_trunk: bool = False,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.hidden = hidden
        self.encoder = MLPEncoder(in_dim, hidden, encoder_layers, dropout)
        # Per-target query vectors
        self.queries = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(hidden) * 0.02) for name in TARGET_NAMES}
        )
        self.shared_trunk = None
        if shared_trunk:
            self.shared_trunk = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.head_sa = nn.Linear(hidden, 1)
        self.head_rrb = nn.Linear(hidden, 1)
        self.head_c2 = nn.Linear(hidden, 3)  # thresholds >=1,2,3
        self.head_b1 = nn.Linear(hidden, 1)
        self.head_b12 = nn.Linear(hidden, 2)  # thresholds >=1,2

    def encode_tasks(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> (B, T, H)"""
        b, t, f = x.shape
        e = self.encoder(x.reshape(b * t, f))
        return e.reshape(b, t, self.hidden)

    def attend(
        self, enc: torch.Tensor, target: str, task_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return context (B,H) and attention (B,T)."""
        # enc: (B, T, H)
        q = self.queries[target]  # (H,)
        scores = torch.einsum("bth,h->bt", enc, q) / math.sqrt(self.hidden)
        if task_mask is not None:
            scores = scores.masked_fill(~task_mask, -1e9)
        alpha = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bt,bth->bh", alpha, enc)
        if self.shared_trunk is not None:
            ctx = self.shared_trunk(ctx)
        return ctx, alpha

    def forward(
        self,
        x: torch.Tensor,
        *,
        task_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode_tasks(x)
        out: dict[str, torch.Tensor] = {}
        attentions: dict[str, torch.Tensor] = {}
        for name, head in (
            ("SA", self.head_sa),
            ("RRB", self.head_rrb),
            ("C2", self.head_c2),
            ("B1", self.head_b1),
            ("B12", self.head_b12),
        ):
            ctx, alpha = self.attend(enc, name, task_mask=task_mask)
            attentions[name] = alpha
            pred = head(ctx)
            if name in ("SA", "RRB", "B1"):
                pred = pred.squeeze(-1)
            out[name] = pred
        out["attention"] = attentions
        return out
