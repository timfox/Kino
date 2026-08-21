"""Configuration for HD LoRA attention theory stub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoraHdAttnConfig:
    """High-dimensional limit parameters (Eq. 7) and regularization."""

    D: int = 150
    T: int = 3
    kappa0: float = 1.0
    kappa: float = 1.0
    alpha: float = 0.1
    alpha_prime: float = 3.0
    delta: float = 0.5
    lam: float = 0.01
    lam_prime: float = 0.05
    e: int = 0  # 0 independent fine-tune samples, 1 reused sequences
    sigma: str = "softmax"  # softmax | identity
    beta: float = 1.0  # frozen-map recalibration scale on W

    @property
    def P0(self) -> int:
        return max(1, int(round(self.kappa0 * self.D)))

    @property
    def P(self) -> int:
        return max(1, int(round(self.kappa * self.D)))

    @property
    def N(self) -> int:
        return max(1, int(round(self.alpha * self.D * self.D)))

    @property
    def N_prime(self) -> int:
        return max(1, int(round(self.alpha_prime * self.D)))

    @property
    def Q0(self) -> float:
        return 1.0 + self.kappa0

    @property
    def q0(self) -> float:
        return 1.0
