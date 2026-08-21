"""CC and adversarial reward functions (Eq. 1–7)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.cclab.config import CCLabConfig


def delay_factor(
    rtt: Tensor,
    rtt_min: Tensor,
    *,
    gamma: float = 1.0,
) -> Tensor:
    """Eq. (2): delay-related factor D_t."""
    threshold = gamma * rtt_min
    return torch.where(threshold < rtt, gamma * rtt_min / rtt.clamp(min=1e-9), torch.ones_like(rtt))


def cc_reward(
    throughput: Tensor,
    loss_rate: Tensor,
    rtt: Tensor,
    rtt_min: Tensor,
    bmax: Tensor,
    *,
    cfg: CCLabConfig | None = None,
) -> Tensor:
    """Eq. (1): learning-based CC reward R_t = T_t - (λ L_t / B_max) · D_t."""
    cfg = cfg or CCLabConfig()
    dt = delay_factor(rtt, rtt_min, gamma=cfg.delay_margin_gamma)
    penalty = cfg.loss_penalty_lambda * loss_rate / bmax.clamp(min=1e-9) * dt
    return throughput - penalty


def naive_adversarial_reward(cc_r: Tensor) -> Tensor:
    """Eq. (3): R^adv_t = -R_t (naive design)."""
    return -cc_r


def queuing_delay(rtt: Tensor, rtt_min: Tensor) -> Tensor:
    """Eq. (4): d_t = RTT_t - RTT_min."""
    return (rtt - rtt_min).clamp(min=0.0)


def delay_penalty(
    delays: Tensor,
    *,
    tau: float,
    h: int = 5,
    k: int = 1,
    alpha: float = 1.0,
) -> Tensor:
    """Eq. (5–6): penalty when both averaged and instant delay are below baseline τ."""
    if delays.numel() == 0:
        return torch.tensor(0.0)
    h = min(h, int(delays.numel()))
    k = min(k, int(delays.numel()))
    d_bar = delays[-h:].mean()
    d_tilde = delays[-k:].mean()
    if float(d_bar) < tau and float(d_tilde) < tau:
        return torch.tensor(-alpha)
    return torch.tensor(0.0)


def improved_adversarial_reward(
    utilization: Tensor,
    delays: Tensor,
    *,
    tau: float,
    cfg: CCLabConfig | None = None,
) -> Tensor:
    """Eq. (7): R^env_t = -U_t + R^delay_t."""
    cfg = cfg or CCLabConfig()
    u = utilization if utilization.ndim == 0 else utilization.mean()
    r_delay = delay_penalty(
        delays,
        tau=tau,
        h=cfg.delay_history_h,
        k=cfg.delay_instant_k,
        alpha=cfg.delay_penalty_alpha,
    )
    return -u + r_delay


def utilization_degradation_pct(clean_util: float, adv_util: float) -> float:
    """Percent drop in bandwidth utilization under attack."""
    if clean_util <= 0:
        return 0.0
    return max(0.0, (clean_util - adv_util) / clean_util * 100.0)
