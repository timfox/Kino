"""P-MLIP model wrapper and config."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.pmlip.egnn import PEGNN


@dataclass
class PMLIPConfig:
    hidden: int = 64
    layers: int = 4
    noise_dim: int = 32
    k_train: int = 10
    k_val: int = 50
    k_test: int = 100


class OrbStub(nn.Module):
    """Lightweight Orb-v3 stand-in for silica fine-tuning demos."""

    def __init__(self, hidden: int = 128, noise_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.energy_head = nn.Linear(hidden, 1)
        self.force_head = nn.Linear(hidden, 3)
        from ltx_trainer.pmlip.noise import NoiseGenerator, PerturbedMLP

        self.noise_gen = NoiseGenerator(noise_dim)
        self.node_mlp = PerturbedMLP(hidden, hidden, hidden, noise_dim)

    def forward_once(self, pos: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(pos)
        h = self.node_mlp(h, z)
        energy = self.energy_head(h).sum()
        forces = self.force_head(h)
        return energy, forces

    def forward(self, pos: Tensor, *, k: int = 10) -> tuple[Tensor, Tensor]:
        energies, forces = [], []
        for _ in range(k):
            z = self.noise_gen.sample()[0]
            e, f = self.forward_once(pos, z)
            energies.append(e)
            forces.append(f)
        return torch.stack(energies), torch.stack(forces)


class PerturbedMLIP(nn.Module):
    """Unified P-MLIP wrapper: P-EGNN or P-Orb backends."""

    def __init__(self, cfg: PMLIPConfig | None = None, *, backend: str = "egnn") -> None:
        super().__init__()
        self.cfg = cfg or PMLIPConfig()
        self.backend = backend
        if backend == "egnn":
            self.model = PEGNN(self.cfg.hidden, self.cfg.layers, self.cfg.noise_dim)
        elif backend == "orb":
            self.model = OrbStub(self.cfg.hidden, self.cfg.noise_dim)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def forward(self, *args, k: int | None = None, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        k = k or self.cfg.k_train
        return self.model(*args, k=k, **kwargs) if self.backend == "egnn" else self.model(args[0], k=k)
