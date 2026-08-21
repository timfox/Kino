"""TimesFM CPS attack detection config (arXiv:2606.06347)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TimesFMAttackConfig:
    paper_arxiv: str = "arXiv:2606.06347"
    packages: tuple[str, ...] = (
        "chi2",
        "plant",
        "observer",
        "attacks",
        "surrogate",
        "detector",
        "simulation",
        "metrics",
    )


@dataclass(frozen=True)
class MassSpringConfig:
    """Section IV-A undamped mass-spring (LTI)."""

    omega: float = 0.3
    dt: float = 1.0
    sigma_v: float = 0.01
    steps: int = 150
    seed: int = 0
    # primary χ² detector
    alpha_p: float = 0.005
    observer_poles: tuple[complex, complex] = (0.5 + 0.1j, 0.5 - 0.1j)
    # Algorithm 1 secondary detector
    context_length: int = 50
    warmup_length: int = 51
    clean_length: int = 20
    alpha_s: float = 0.005
    buffer_theta: float = 0.8
    # replay attack
    replay_k0: int = 50
    replay_k1: int = 70
    replay_ka: int = 71
    # stealthy attack (Theorem 1)
    stealthy_ka: int = 81
    stealthy_k2: int = 100
    delta_tau_p: float = 0.385
    impact_weight: tuple[float, ...] = (1.0, 0.0)


@dataclass(frozen=True)
class IEEE14BusAnchors:
    """Literature anchors — not re-simulated in this stub."""

    buses: int = 14
    lines: int = 20
    m_outputs: int = 34
    alpha_p: float = 0.001
    context_length: int = 100
    warmup_length: int = 110
    clean_length: int = 200
    replay_k0: int = 60
    replay_k1: int = 70
    replay_ka: int = 71
    stealthy_ka: int = 111
    stealthy_k2: int = 161
    delta_tau_p: float = 5.87
    sigma_omega: float = 0.035
    sigma_f: float = 0.050
