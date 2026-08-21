"""Latent consistency diffusion training and few-step inference (toy NumPy)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.aircast_sr.config import AirCastSRConfig
from ltx_trainer.aircast_sr.schedule import add_noise, lcm_inference_timesteps
from ltx_trainer.aircast_sr.unet3d import UNet3DWeights, build_toy_unet3d, unet3d_forward


def build_denoiser_input(
    noisy_target: np.ndarray,
    conditioning: np.ndarray,
    *,
    lead_time_plane: float | None = None,
) -> np.ndarray:
    """Concatenate (7, T, H, W) + (20, T, H, W) [+ optional 1 lead plane] → 28 channels."""
    parts = [noisy_target, conditioning]
    if lead_time_plane is not None:
        plane = np.full((1, *noisy_target.shape[1:]), lead_time_plane, dtype=np.float64)
        parts.append(plane)
    return np.concatenate(parts, axis=0)


def training_mse_loss(
    target: np.ndarray,
    conditioning: np.ndarray,
    weights: UNet3DWeights,
    *,
    timestep: int,
    rng: np.random.Generator,
    cfg: AirCastSRConfig | None = None,
) -> float:
    cfg = cfg or AirCastSRConfig()
    noise = rng.standard_normal(target.shape)
    noisy = add_noise(target, noise, timestep, num_timesteps=cfg.training_timesteps)
    den_in = build_denoiser_input(noisy, conditioning, lead_time_plane=0.5)
    pred = unet3d_forward(den_in, weights, groups=cfg.group_norm_groups)
    return float(np.mean((pred - noise) ** 2))


def lcm_sample(
    conditioning: np.ndarray,
    weights: UNet3DWeights,
    *,
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    cfg: AirCastSRConfig | None = None,
) -> np.ndarray:
    """Generate normalized target tensor (7, T, H, W) from noise."""
    cfg = cfg or AirCastSRConfig()
    c_out, t, h, w = 7, *shape
    z = rng.standard_normal((c_out, t, h, w))
    steps = lcm_inference_timesteps(cfg.lcm_inference_steps, num_train_steps=cfg.training_timesteps)
    for t_idx in steps:
        den_in = build_denoiser_input(z, conditioning, lead_time_plane=float(t_idx) / cfg.training_timesteps)
        eps = unet3d_forward(den_in, weights, groups=cfg.group_norm_groups)
        z = z - 0.1 * eps
    return np.clip(z, 0.0, 1.0)


def fresh_denoiser(seed: int = 0, *, smoke: bool = True) -> UNet3DWeights:
    return build_toy_unet3d(AirCastSRConfig(), seed=seed, smoke=smoke)
