"""GCP-ISM high-dimensional image-source RIRs (DAFx26 / arXiv:2606.04358)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GCPIsmConfig:
    paper_arxiv: str = "arXiv:2606.04358"
    venue: str = "DAFx26, Cambridge MA, 1–4 Sep 2026"
    title: str = (
        "Gauss Circle Lattices with Geometric Convolutions for Synthesizing "
        "High Dimensional Image-Source Room Impulse Responses"
    )
    github: str = "https://github.com/yluo1/GCP-ISM"
    c_sound: float = 343.0
    sample_rate_hz: int = 48_000

    # Fig. 5 demo room (integer Z^N)
    source: tuple[int, ...] = (1, 0, 1)
    receiver: tuple[int, ...] = (2, 1, 1)
    orthotope: tuple[int, ...] = (5, 4, 3)
    gamma_plus: tuple[float, ...] = (0.93, 0.8, 0.9)
    gamma_minus: tuple[float, ...] = (0.72, 0.78, 0.8)
    duration_s: float = 0.3
    lambda_scale: int = 1
    lanczos_alpha: int = 10

    # Complexity claims
    direct_ism_big_o: str = "O(k^N)"
    gcp_ism_big_o: str = "O(N k^2 log k)"

    # Fig. 9: doubling λ → ~12 dB NMSE drop
    lambda_nmse_drop_db: float = 12.0

    demo_k_max: int = 64
    demo_n_dims: int = 3
