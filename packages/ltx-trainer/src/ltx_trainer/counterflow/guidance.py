"""Two-phase decomposed guidance for flow-matching VT2A (Eq. 1–2)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

Velocity = np.ndarray


class VelocityProvider(Protocol):
    """Toy interface: v(Z, c_vid, c_txt) -> velocity."""

    def __call__(
        self,
        z: Velocity,
        *,
        use_video: bool,
        text_mode: str,  # "null" | "target" | "source"
    ) -> Velocity: ...


def phase1_velocity(
    provider: VelocityProvider,
    z: Velocity,
    *,
    w_vid: float = 3.0,
    w_txt: float = 5.0,
) -> Velocity:
    """
    Eq. (1) — Phase 1: video on, decomposed text (target vs source).

    v^(1) = v(∅,∅) + w_vid·(v(vid,∅)−v(∅,∅)) + w_txt·(v(∅,tar)−v(∅,src))
    """
    v_uncond = provider(z, use_video=False, text_mode="null")
    v_vid = provider(z, use_video=True, text_mode="null")
    v_tar = provider(z, use_video=False, text_mode="target")
    v_src = provider(z, use_video=False, text_mode="source")
    return v_uncond + w_vid * (v_vid - v_uncond) + w_txt * (v_tar - v_src)


def phase2_velocity(
    provider: VelocityProvider,
    z: Velocity,
    *,
    w_cfg: float = 4.5,
) -> Velocity:
    """
    Eq. (2) — Phase 2: video off, target vs source text CFG.

    v^(2) = v(∅,∅) + w_cfg·(v(∅,tar)−v(∅,src))
    """
    v_uncond = provider(z, use_video=False, text_mode="null")
    v_tar = provider(z, use_video=False, text_mode="target")
    v_src = provider(z, use_video=False, text_mode="source")
    return v_uncond + w_cfg * (v_tar - v_src)


def vanilla_cfg_velocity(
    provider: VelocityProvider,
    z: Velocity,
    *,
    w: float = 4.5,
) -> Velocity:
    """Vanilla CFG baseline: v(∅,∅) + w·(v(vid,tar) − v(∅,∅))."""
    v_uncond = provider(z, use_video=False, text_mode="null")
    v_joint = provider(z, use_video=True, text_mode="target")
    return v_uncond + w * (v_joint - v_uncond)


def euler_sample_counterflow(
    provider: VelocityProvider,
    z0: Velocity,
    *,
    num_steps: int = 25,
    transition_step: int = 17,
    w_vid: float = 3.0,
    w_txt: float = 5.0,
    w_cfg: float = 4.5,
    dt: float = 1.0 / 25.0,
) -> list[Velocity]:
    """Deterministic Euler ODE integration with phase switch at N_trans."""
    states = [z0.copy()]
    z = z0.copy()
    for i in range(num_steps):
        if i < transition_step:
            v = phase1_velocity(provider, z, w_vid=w_vid, w_txt=w_txt)
        else:
            v = phase2_velocity(provider, z, w_cfg=w_cfg)
        z = z + dt * v
        states.append(z.copy())
    return states
