"""SKILD configuration (schedules + SNR protocol from arXiv:2605.26032)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkildScheduleConfig:
    """Linear schedule (ImageNet SR defaults from paper supplement E)."""

    family: str = "linear"
    num_steps: int = 1000
    theta: float = 9.0
    lambda_i: float = 1132.9352
    lambda_f: float = 550.8723
    kc: float = 0.0
    alpha_floor: float = 1e-6


@dataclass
class SkildConfig:
    """Runtime SKILD settings."""

    schedule: SkildScheduleConfig = field(default_factory=SkildScheduleConfig)
    snr_threshold: float = 0.1
    """SNR per mode for effective-resolution cutoff (paper ImageNet SR protocol)."""
    channel_wise: bool = True
    """Estimate / apply spectrum per RGB channel."""
