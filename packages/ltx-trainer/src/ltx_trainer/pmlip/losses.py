"""CRPS training losses for energy and forces (Sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.pmlip.crps import fair_crps_multivariate


@dataclass
class PMLIPLossConfig:
    k_train: int = 10


class PMLIPLoss:
    def __init__(self, cfg: PMLIPLossConfig | None = None) -> None:
        self.cfg = cfg or PMLIPLossConfig()

    def __call__(self, samples: Tensor, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        """``samples``: ``(K, ...)``, ``target``: same shape without K dim."""
        flat_s = samples.reshape(samples.shape[0], -1)
        flat_t = target.reshape(-1)
        loss = fair_crps_multivariate(flat_s, flat_t)
        mse = ((samples.mean(0) - target) ** 2).mean()
        return loss, {"loss_crps": float(loss.detach()), "loss_mse_mean": float(mse.detach())}


def joint_energy_forces_crps(
    energy_samples: Tensor,
    force_samples: Tensor,
    energy_target: Tensor,
    force_target: Tensor,
    *,
    n_atoms: int,
) -> tuple[Tensor, dict[str, float]]:
    """P-Orb convention: L_CRPS(E/N) + L_CRPS(F) (Sec. 3.2)."""
    e_per_atom = energy_samples / n_atoms
    e_tgt = energy_target / n_atoms
    l_e = fair_crps_multivariate(e_per_atom.reshape(energy_samples.shape[0], -1), e_tgt.reshape(-1))
    l_f = fair_crps_multivariate(
        force_samples.reshape(force_samples.shape[0], -1),
        force_target.reshape(-1),
    )
    total = l_e + l_f
    return total, {"loss_energy_crps": float(l_e.detach()), "loss_forces_crps": float(l_f.detach())}
