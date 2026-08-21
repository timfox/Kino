"""Latitude-based adaptive quality parameter (Sec. IV-A, Eq. 8–12)."""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor

from ltx_trainer.nvc_erp_qpa.config import NvcErpQpaConfig, log_lambda_span


def latitude_from_row(row: int, height: int) -> float:
    """Latitude φ ∈ [-π/2, π/2] for ERP row index (top = +π/2)."""
    if height <= 1:
        return 0.0
    # row 0 → φ = π/2, row H-1 → φ = -π/2
    t = row / (height - 1)
    return math.pi / 2 - math.pi * t


def delta_q_phi(
    phi: Union[float, Tensor],
    *,
    q_num: int,
    lambda_min: float,
    lambda_max: float,
) -> Union[float, Tensor]:
    """Eq. (10): Δqφ = (q_num-1)·ln(cos φ) / (ln λ_max - ln λ_min)."""
    span = math.log(lambda_max) - math.log(lambda_min)
    cos_phi = torch.cos(phi) if isinstance(phi, Tensor) else math.cos(phi)
    if isinstance(cos_phi, Tensor):
        cos_phi = cos_phi.clamp(min=1e-6)
    else:
        cos_phi = max(cos_phi, 1e-6)
    return (q_num - 1) * (torch.log(cos_phi) if isinstance(cos_phi, Tensor) else math.log(cos_phi)) / span


def mean_delta_q(*, q_num: int, lambda_min: float, lambda_max: float) -> float:
    """Eq. (12): mean of Δqφ over φ ∈ [-π/2, π/2]."""
    span = math.log(lambda_max) - math.log(lambda_min)
    return -(q_num - 1) * math.log(2) / span


def adaptive_q_tilde(
    q0: float,
    phi: Union[float, Tensor],
    *,
    q_num: int,
    lambda_min: float,
    lambda_max: float,
) -> Union[float, Tensor]:
    """Eq. (11): q̃φ = q0 + Δqφ − Δ̄q."""
    dq = delta_q_phi(phi, q_num=q_num, lambda_min=lambda_min, lambda_max=lambda_max)
    dq_bar = mean_delta_q(q_num=q_num, lambda_min=lambda_min, lambda_max=lambda_max)
    return q0 + dq - dq_bar


def q_to_lambda(q: Union[float, Tensor], cfg: NvcErpQpaConfig) -> Union[float, Tensor]:
    """Eq. (2): λ from integer quality index."""
    if cfg.q_num <= 1:
        return cfg.lambda_min
    t = q / (cfg.q_num - 1)
    ln_lam = math.log(cfg.lambda_min) + t * log_lambda_span(cfg)
    if isinstance(q, Tensor):
        return torch.exp(torch.tensor(ln_lam, device=q.device, dtype=q.dtype))
    return math.exp(ln_lam)


def latitude_q_map(height: int, cfg: NvcErpQpaConfig) -> Tensor:
    """Per-row adaptive q̃φ, shape (H,)."""
    rows = torch.arange(height, dtype=torch.float32)
    phis = torch.tensor(
        [latitude_from_row(int(r.item()), height) for r in rows],
        dtype=torch.float32,
    )
    q_map = torch.stack(
        [
            torch.as_tensor(
                adaptive_q_tilde(
                    cfg.q0,
                    float(phi),
                    q_num=cfg.q_num,
                    lambda_min=cfg.lambda_min,
                    lambda_max=cfg.lambda_max,
                ),
                dtype=torch.float32,
            )
            for phi in phis
        ]
    )
    return q_map.clamp(0.0, float(cfg.q_num - 1))
