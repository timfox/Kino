"""Training-step helpers for PiD FM + adapter (Sec. 3.3)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.pid.config import PiDConfig
from ltx_trainer.pid.decode import PiDDecoder, training_loss
from ltx_trainer.pid.flow import corrupt_latent, interpolate_state, sample_noise_like


def sample_latent_sigma(cfg: PiDConfig | None = None) -> float:
    cfg = cfg or PiDConfig()
    return float(torch.rand(1).item() * cfg.sigma_max)


def prepare_training_batch(
    x0: Tensor,
    *,
    latent: Tensor,
    cfg: PiDConfig | None = None,
) -> dict[str, Tensor]:
    """Build x_t, noisy latent, and targets for Eq. (10)."""
    cfg = cfg or PiDConfig()
    t = float(torch.rand(1).item())
    eps = sample_noise_like(x0)
    x_t = interpolate_state(x0, eps, t)
    sigma = sample_latent_sigma(cfg)
    z_noisy = corrupt_latent(latent, sigma)
    return {"x_t": x_t, "x0": x0, "epsilon": eps, "t": torch.tensor(t), "z_noisy": z_noisy, "sigma": torch.tensor(sigma)}


def pid_training_loss(
    batch: dict[str, Tensor],
    *,
    cfg: PiDConfig | None = None,
    latent_channels: int = 4,
) -> dict[str, Any]:
    cfg = cfg or PiDConfig()
    decoder = PiDDecoder(cfg=cfg, latent_channels=latent_channels)
    sigma = float(batch["sigma"].item())
    v_pred = decoder.velocity(batch["x_t"], float(batch["t"].item()))
    loss = training_loss(
        v_pred,
        batch["x0"],
        batch["epsilon"],
        latent=batch.get("z_noisy"),
        sigma_latent=sigma,
    )
    return {"loss": loss, "sigma_latent": sigma, "t": float(batch["t"].item())}
