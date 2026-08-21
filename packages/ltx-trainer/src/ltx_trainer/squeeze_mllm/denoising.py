"""Multi-stage timestep-aware denoising masks (Eq. 5, arXiv:2605.26111)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.squeeze_mllm.config import SqueezeMLLMConfig


@dataclass(frozen=True)
class StageMasks:
    use_mllm: bool
    use_vae: bool


def stage_masks(t: float | Tensor, cfg: SqueezeMLLMConfig | None = None) -> StageMasks:
    """Return active branches for normalized timestep ``t`` in [0, 1] (1=noise, 0=data)."""
    cfg = cfg or SqueezeMLLMConfig()
    if isinstance(t, Tensor):
        t_val = float(t.item())
    else:
        t_val = float(t)
    if t_val >= cfg.tau1:
        return StageMasks(use_mllm=True, use_vae=False)
    if t_val >= cfg.tau2:
        return StageMasks(use_mllm=True, use_vae=True)
    return StageMasks(use_mllm=False, use_vae=True)


def mask_schedule(
    num_steps: int = 25,
    cfg: SqueezeMLLMConfig | None = None,
) -> list[dict[str, bool | float]]:
    """Discrete schedule from noise (t=1) to data (t=0) for visualization."""
    cfg = cfg or SqueezeMLLMConfig()
    out: list[dict[str, bool | float]] = []
    for i in range(num_steps):
        t = 1.0 - i / max(1, num_steps - 1)
        m = stage_masks(t, cfg)
        out.append(
            {
                "t": t,
                "use_mllm": m.use_mllm,
                "use_vae": m.use_vae,
                "stage": (
                    "early_mllm"
                    if m.use_mllm and not m.use_vae
                    else "middle_both"
                    if m.use_mllm and m.use_vae
                    else "late_vae"
                ),
            }
        )
    return out
