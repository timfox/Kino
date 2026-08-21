"""Branch energy localization simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.branch_energy.constants import F0_HZ


@dataclass
class BranchEnergyConfig:
    f0_hz: float = F0_HZ
    num_periods: float = 5.0
    fs_hz: float = 20_000.0
    tau_residual: float = 1e-12
