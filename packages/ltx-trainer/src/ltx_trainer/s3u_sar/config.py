"""S³U-SAR demo configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class S3USarConfig:
    temperature: float = 1.0
    alpha: float = 0.8
    beta: float = 0.15
    mu: float = 0.4
    gamma_topo: float = 1.0
    lambda_high: float = 2.0
    lambda_low: float = 0.5
