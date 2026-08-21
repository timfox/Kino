"""Hybrid UNet + FNO architecture stub — Sec. III-A."""

from __future__ import annotations

from typing import Any

from ltx_trainer.flood_physics.config import FloodPhysicsConfig
from ltx_trainer.flood_physics.fno import fno_forward


def unet_local_path(flat: list[float], *, channels: int = 64) -> list[float]:
    """Encoder-decoder local feature path (1-D stub with skip residual)."""
    local = [x * 1.05 + 0.01 for x in flat]
    return [0.5 * (a + b) for a, b in zip(flat, local)]


def fuse_unet_fno(unet_feat: list[float], fno_feat: list[float]) -> list[float]:
    """Multi-resolution fusion via concatenation + residual."""
    return [0.5 * (u + f) for u, f in zip(unet_feat, fno_feat)]


def mlp_head(fused: list[float]) -> dict[str, list[float]]:
    """Predict h, u, v and extent probability per cell."""
    extent_p = [min(max(0.5 + 0.1 * x, 0.0), 1.0) for x in fused]
    depth = [max(0.0, 0.3 + 0.05 * x) for x in fused]
    u_vel = [0.02 * x for x in fused]
    v_vel = [0.01 * x for x in fused]
    return {"extent_p": extent_p, "depth": depth, "u": u_vel, "v": v_vel}


def hybrid_forward(
    sar: list[float],
    optical: list[float],
    dem: list[float],
    cfg: FloodPhysicsConfig | None = None,
) -> dict[str, Any]:
    """Fuse multi-modal inputs → UNet local + FNO global → MLP hydraulics."""
    cfg = cfg or FloodPhysicsConfig()
    flat = [0.4 * s + 0.4 * o + 0.2 * d for s, o, d in zip(sar, optical, dem)]
    unet_out = unet_local_path(flat, channels=cfg.unet_channels)
    fno_out = fno_forward(flat, modes=cfg.fno_modes)
    fused = fuse_unet_fno(unet_out, fno_out)
    head = mlp_head(fused)
    return {"fused": fused, **head}
