"""Wiring temporal encoder → disentangled embedder → trajectory head (Fig. 6)."""

from __future__ import annotations

from torch import Tensor

from ltx_trainer.deltacam.style import DisentangledStyleEmbedder, StyleTrajectoryHead
from ltx_trainer.deltacam.temporal import TemporalStyleEncoder


def encode_style_trajectory(
    patch_seq: Tensor,
    *,
    temporal: TemporalStyleEncoder,
    embedder: DisentangledStyleEmbedder,
    head: StyleTrajectoryHead,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run shared temporal encoding then per-frame content/style and τ̂ (Sec. 3.3).

    ``patch_seq``: ``[B, T, D]`` patch-level sequence (e.g. would come from a frozen ViT).
    Returns ``(tau_hat, z_content, z_style)`` each ``[B, T, …]``.
    """
    if patch_seq.dim() != 3:
        raise ValueError(f"Expected [B, T, D], got {tuple(patch_seq.shape)}")
    h = temporal(patch_seq)
    b, t, d = h.shape
    flat = h.reshape(b * t, d)
    z_c, z_s = embedder(flat)
    tau_hat = head(z_s)
    num_p = tau_hat.shape[-1]
    d_c, d_s = z_c.shape[-1], z_s.shape[-1]
    return (
        tau_hat.view(b, t, num_p),
        z_c.view(b, t, d_c),
        z_s.view(b, t, d_s),
    )
