"""SDT2I sampler orchestrator."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.sdt2i.config import Sdt2iConfig
from ltx_trainer.sdt2i.mpf import MultiPanFusionStub
from ltx_trainer.sdt2i.mstd import MultiStitchDiffusionStub


class Sdt2iStub(nn.Module):
    def __init__(self, cfg: Sdt2iConfig | None = None, *, variant: str = "mstd") -> None:
        super().__init__()
        self.cfg = cfg or Sdt2iConfig()
        self.variant = variant
        self.mstd = MultiStitchDiffusionStub(self.cfg)
        self.mpf = MultiPanFusionStub(self.cfg)

    def forward(
        self,
        latent: Tensor,
        masks: list[Tensor],
        *,
        steps: int = 4,
    ) -> dict[str, Tensor]:
        z = latent
        for t in range(steps):
            if self.variant == "mpf":
                z = self.mpf(z, masks, bootstrap_step=t)
            else:
                z = self.mstd(z, masks, step=t)
        certainty = torch.sigmoid(z.mean(dim=1, keepdim=True))
        return {"latent": z, "certainty": certainty}
