"""λ(t) schedules and per-mode ᾱ_n(k)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.skild.config import SkildScheduleConfig


class SkildSchedule:
    def __init__(self, cfg: SkildScheduleConfig, *, k_grid: np.ndarray):
        self.cfg = cfg
        self.k_grid = np.asarray(k_grid, dtype=np.float64)
        self._k_eff = np.maximum(self.k_grid, float(cfg.kc))

    def lambda_at(self, t: float) -> float:
        c = self.cfg
        if c.family == "log_linear":
            return float(t * (10 ** (c.lambda_i + (c.lambda_f - c.lambda_i) * t)))
        # linear (paper Eq. D.49)
        denom = c.lambda_f * (1.0 - t) + c.lambda_i
        return float(c.theta * t / (denom * denom + 1e-12))

    def alpha_bar(self, step: int) -> np.ndarray:
        """``ᾱ_n(k) = exp(-k² λ(t_n))`` for discrete step ``n`` in ``[0, N-1]``."""
        n = int(step)
        t = n / max(self.cfg.num_steps - 1, 1)
        lam = self.lambda_at(t)
        return np.exp(-(self._k_eff * self._k_eff) * lam)

    def alpha_step(self, step: int) -> np.ndarray:
        ab_prev = self.alpha_bar(max(step - 1, 0))
        ab = self.alpha_bar(step)
        ratio = ab / np.maximum(ab_prev, self.cfg.alpha_floor)
        return np.clip(ratio, self.cfg.alpha_floor, 1.0)

    def beta_step(self, step: int) -> np.ndarray:
        return 1.0 - self.alpha_step(step)

    def snr(self, step: int) -> np.ndarray:
        ab = self.alpha_bar(step)
        return ab / np.maximum(1.0 - ab, 1e-12)
