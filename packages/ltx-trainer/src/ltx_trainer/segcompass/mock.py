"""SegCompass SAE smoke."""

from __future__ import annotations

from typing import Any


def toy_sparse_activations(*, n_active: int, dim: int) -> list[float]:
    h = [0.0] * dim
    for i in range(n_active):
        h[i] = 1.0
    return h


def toy_reasoning_instruction() -> str:
    return "segment the object mentioned in the reasoning chain"


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.segcompass.sae import sae_reconstruction_loss, sparsity_fraction, support_indices

    h = toy_sparse_activations(n_active=8, dim=256)
    idx = support_indices(h)
    z_norm = sum(v * v for v in h) ** 0.5
    loss = sae_reconstruction_loss(z_norm, z_norm, sum(abs(x) for x in h))
    sp = sparsity_fraction(len(idx), len(h))
    return {
        "instruction_words": len(toy_reasoning_instruction().split()),
        "active_features": len(idx),
        "recon_loss": round(loss, 4),
        "sparsity": round(sp, 4),
    }
