"""Hidden number linear probing — Eq. (1), § 2.3."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StonePresenceProbe(nn.Module):
    """Binary linear probe f_probe: R^d -> {0,1} per patch."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, z: Tensor) -> Tensor:
        return (torch.sigmoid(self.linear(z)).squeeze(-1) > 0.5).float()

    def hidden_number(self, z: Tensor) -> Tensor:
        """N_H = sum_i f_probe(z_i) — Eq. (1)."""
        return self.forward(z).sum(dim=-1)


def train_probe_on_id(
    probe: StonePresenceProbe,
    samples: list[tuple[Tensor, Tensor]],
    *,
    steps: int = 50,
    lr: float = 0.05,
) -> float:
    """Train probe on ID patches only (N <= 49)."""
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    last = 0.0
    for _ in range(steps):
        total = probe.linear.weight.new_zeros(())
        for emb, labels in samples:
            logits = probe.linear(emb).squeeze(-1)
            total = total + loss_fn(logits, labels)
        opt.zero_grad()
        (total / len(samples)).backward()
        opt.step()
        last = float(total.item() / len(samples))
    return last


def steering_accuracy(
    probe: StonePresenceProbe,
    emb: Tensor,
    ng: int,
    k: int,
    *,
    predicted_after_mask: int,
) -> bool:
    """Causal mask intervention: N'_P == N_G - k — § D.1.1."""
    return predicted_after_mask == ng - k
