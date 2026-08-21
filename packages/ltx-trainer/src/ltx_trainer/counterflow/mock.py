"""Toy velocity provider for CounterFlow smoke tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.counterflow.guidance import euler_sample_counterflow, phase1_velocity, phase2_velocity

Velocity = np.ndarray


class LinearVelocityProvider:
    """Deterministic linear toy provider implementing ``VelocityProvider``."""

    def __init__(self, dim: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._bias = rng.standard_normal(dim)

    def __call__(
        self,
        z: Velocity,
        *,
        use_video: bool,
        text_mode: str,
    ) -> Velocity:
        mode_shift = {"null": 0.0, "target": 0.15, "source": -0.1}.get(text_mode, 0.0)
        vid_scale = 0.1 if use_video else 0.0
        return self._bias + vid_scale * z + mode_shift


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    z = np.zeros(8)
    provider = LinearVelocityProvider(dim=8, seed=seed)
    v1 = phase1_velocity(provider, z)
    v2 = phase2_velocity(provider, z)
    traj = euler_sample_counterflow(provider, z, num_steps=5, transition_step=3)
    return {
        "phase1_norm": round(float(np.linalg.norm(v1)), 4),
        "phase2_norm": round(float(np.linalg.norm(v2)), 4),
        "trajectory_len": len(traj),
    }
