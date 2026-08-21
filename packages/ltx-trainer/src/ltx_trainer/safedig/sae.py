"""Sparse autoencoder + decoder-only transfer (Eq. 4–5, 12–15)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ltx_trainer.safedig.config import SafeDIGConfig


class SparseAutoencoder(nn.Module):
    """SAE: z = E(a), â = D(z) with L1 sparsity — Eq. (4–5)."""

    def __init__(self, dim: int, *, expansion: int = 16, sparsity_lambda: float = 0.03125) -> None:
        super().__init__()
        hidden = dim * expansion
        self.encoder = nn.Linear(dim, hidden, bias=True)
        self.decoder = nn.Linear(hidden, dim, bias=True)
        self.sparsity_lambda = sparsity_lambda

    def encode(self, a: Tensor) -> Tensor:
        return self.encoder(a)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, a: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(a)
        return z, self.decode(z)

    def reconstruction_loss(self, a: Tensor) -> Tensor:
        z, a_hat = self.forward(a)
        mse = (a - a_hat).pow(2).mean()
        l1 = z.abs().mean()
        return mse + self.sparsity_lambda * l1


def contrast_activation(a_safe: Tensor, a_harm: Tensor) -> Tensor:
    """Positive-minus-negative safety contrast — Eq. (6)."""
    return a_safe - a_harm


def train_source_sae(
    sae: SparseAutoencoder,
    activations: Tensor,
    *,
    lr: float = 1e-4,
    steps: int = 10,
) -> float:
    """Source-domain SAE pretrain stub — Eq. (13)."""
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    sae.train()
    last = 0.0
    for _ in range(steps):
        opt.zero_grad()
        loss = sae.reconstruction_loss(activations)
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def decoder_only_transfer(
    sae: SparseAutoencoder,
    target_activations: Tensor,
    *,
    replay_activations: Tensor | None = None,
    replay_ratio: float = 0.2,
    lr: float = 5e-5,
    steps: int = 5,
    mu_stable: float = 0.01,
) -> float:
    """Freeze encoder ω_s, adapt decoder η_t — Eq. (14–15)."""
    for p in sae.encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(sae.decoder.parameters(), lr=lr)
    sae.train()
    last = 0.0
    n_tgt = target_activations.shape[0]
    n_rep = 0 if replay_activations is None else replay_activations.shape[0]
    for _ in range(steps):
        if replay_activations is not None and n_rep > 0 and replay_ratio > 0:
            n_r = max(1, int(n_tgt * replay_ratio))
            idx_r = torch.randint(0, n_rep, (n_r,))
            idx_t = torch.randint(0, n_tgt, (n_tgt - n_r,))
            batch = torch.cat([target_activations[idx_t], replay_activations[idx_r]], dim=0)
        else:
            batch = target_activations
        opt.zero_grad()
        z = sae.encode(batch)
        a_hat = sae.decode(z)
        recon = (batch - a_hat).pow(2).mean()
        stable = mu_stable * (a_hat - batch.detach()).pow(2).mean()
        loss = recon + stable
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def default_hooks(cfg: SafeDIGConfig | None = None) -> list:
    from ltx_trainer.safedig.config import InterventionHook

    _ = cfg
    return [
        InterventionHook("text_encoder", "Ltext", 4096, "text_encoder"),
        InterventionHook("double_stream_block_18", "Lbind", 3072, "double_transformer_blocks.18"),
        InterventionHook("single_stream_block_37", "Lrender", 3072, "single_transformer_blocks.37"),
    ]
