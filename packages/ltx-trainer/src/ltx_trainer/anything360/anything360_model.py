"""360Anything image/video stub (Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.anything360.circular_latent import StubVAEEncoder
from ltx_trainer.anything360.conditioning import (
    GeometryFreeDiTStub,
    sequence_concat_conditioning,
    unpatchify_tokens,
)
from ltx_trainer.anything360.config import Anything360Config


class Anything360Stub(nn.Module):
    def __init__(self, cfg: Anything360Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Anything360Config()
        c = self.cfg.latent_channels
        ps = self.cfg.patch_size
        token_dim = c * ps * ps
        self.vae = StubVAEEncoder(3, c)
        self.dit = GeometryFreeDiTStub(token_dim, depth=2)
        self.token_dim = token_dim

    def encode_perspective(self, x: Tensor) -> Tensor:
        return self.vae(x, circular=False)

    def encode_erp(self, y: Tensor) -> Tensor:
        return self.vae(y, circular=self.cfg.use_circular_latent)

    def forward(
        self,
        pers: Tensor,
        erp: Tensor,
        *,
        t: float = 0.5,
    ) -> dict[str, Tensor]:
        """
        pers: B×3×h×w perspective conditioning
        erp: B×3×H×W target ERP (clean); flow-matching noise added internally
        """
        b = pers.shape[0]
        z_p = self.encode_perspective(pers)
        z_e = self.encode_erp(erp)
        noise = torch.randn_like(z_e)
        z_noisy = (1 - t) * z_e + t * noise

        tokens = sequence_concat_conditioning(z_p, z_noisy, patch_size=self.cfg.patch_size)
        n_pers = patchify_count(z_p, self.cfg.patch_size)
        pred_tokens = self.dit(tokens, n_pers)
        z_pred = unpatchify_tokens(
            pred_tokens, z_e.shape[2], z_e.shape[3], self.cfg.patch_size, self.cfg.latent_channels
        )
        erp_hat = self.vae.decode(z_pred)
        return {
            "z_e": z_e,
            "z_pred": z_pred,
            "noise_target": noise - z_e,
            "erp_recon": F.interpolate(erp_hat, size=erp.shape[-2:], mode="bilinear", align_corners=False),
        }


def patchify_count(z: Tensor, patch_size: int) -> int:
    h, w = z.shape[-2], z.shape[-1]
    return (h // patch_size) * (w // patch_size)
