"""PIU unlearning losses (Sec. 3.2, Eq. 1–4)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.piu.config import PIUConfig


def forget_target_noise(
    eps_frozen_forget: Tensor,
    eps_frozen_anchor: Tensor,
    *,
    eta: float = 1.5,
) -> Tensor:
    """ε̂_forget = ε_θ*(z,t,c_a) − η[ε_θ*(z,t,c_f) − ε_θ*(z,t,c_a)] (Eq. 1)."""
    return eps_frozen_anchor - eta * (eps_frozen_forget - eps_frozen_anchor)


def forget_loss(
    eps_trainable_forget: Tensor,
    eps_target: Tensor,
) -> Tensor:
    """L_forget = E[‖ε_θ(z,t,c_f) − ε̂_forget‖²] (Eq. 2)."""
    return torch.mean((eps_trainable_forget - eps_target) ** 2)


def preserve_loss(
    eps_trainable_retain: Tensor,
    eps_frozen_retain: Tensor,
) -> Tensor:
    """L_preserve = E[‖ε_θ(z,t,c_r) − ε_θ*(z,t,c_r)‖²] (Eq. 3)."""
    return torch.mean((eps_trainable_retain - eps_frozen_retain) ** 2)


def total_piu_loss(
    eps_train_forget: Tensor,
    eps_train_retain: Tensor,
    eps_frozen_forget: Tensor,
    eps_frozen_anchor: Tensor,
    eps_frozen_retain: Tensor,
    *,
    cfg: PIUConfig | None = None,
) -> dict[str, Tensor]:
    """L_total = L_forget + λ L_preserve (Eq. 4)."""
    cfg = cfg or PIUConfig()
    target = forget_target_noise(eps_frozen_forget, eps_frozen_anchor, eta=cfg.negative_guidance_eta)
    l_forget = forget_loss(eps_train_forget, target)
    l_preserve = preserve_loss(eps_train_retain, eps_frozen_retain)
    l_total = l_forget + cfg.preservation_lambda * l_preserve
    return {
        "l_forget": l_forget,
        "l_preserve": l_preserve,
        "loss": l_total,
    }
