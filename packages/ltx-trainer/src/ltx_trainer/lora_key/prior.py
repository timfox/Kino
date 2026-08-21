"""Stage 1 latent watermark prior — Eqs. 7–9."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LatentWatermarkPrior(nn.Module):
    """Message encoder E_ψ and private decoder D_φ over frozen VAE latents."""

    def __init__(self, *, latent_dim: int = 4 * 128 * 128, message_bits: int = 48, hidden: int = 256) -> None:
        super().__init__()
        self.message_bits = message_bits
        self.encoder = nn.Sequential(
            nn.Linear(message_bits, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, message_bits),
        )

    def encode_residual(self, message: Tensor) -> Tensor:
        """E_ψ(m) → δ_z with same shape as z."""
        return self.encoder(message.float())

    def embed(self, z_clean: Tensor, message: Tensor) -> Tensor:
        """z_wm = z + E_ψ(m) (Eq. 7)."""
        delta = self.encode_residual(message)
        return z_clean + delta.view_as(z_clean)

    def decode_message(self, z_wm: Tensor) -> Tensor:
        """m' = D_φ(z) (Eq. 8, latent path)."""
        flat = z_wm.reshape(z_wm.shape[0], -1)
        return torch.sigmoid(self.decoder(flat))


def prior_loss(
    message: Tensor,
    decoded: Tensor,
    *,
    lambda_mse: float = 1.0,
    lambda_lpips: float = 0.1,
    recon_mse: float | None = None,
) -> dict[str, float]:
    """L_prior = L_BCE + λ1 L_MSE + λ2 L_LPIPS (Eq. 9)."""
    bce = float(F.binary_cross_entropy(decoded, message.float()).item())
    mse = recon_mse if recon_mse is not None else 0.0
    lpips = 0.0  # stub: external perceptual net in full pipeline
    total = bce + lambda_mse * mse + lambda_lpips * lpips
    return {"loss": total, "l_bce": bce, "l_mse": mse, "l_lpips": lpips}
