"""WavTTS DiT flow-matching stub on patchified waveforms."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.wavtts.config import WavTTSConfig
from ltx_trainer.wavtts.patchify import patchify_waveform, unpatchify_waveform


class WavTTSStub(nn.Module):
    """Patch embed + shallow Transformer; predicts clean waveform patches."""

    def __init__(self, cfg: WavTTSConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WavTTSConfig()
        f = self.cfg.patch_size
        d = 256
        self.in_proj = nn.Linear(f, d)
        self.ctx_proj = nn.Linear(f, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, batch_first=True, norm_first=True)
        self.dit = nn.TransformerEncoder(layer, num_layers=2)
        self.t_embed = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.out_proj = nn.Linear(d, f)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        x_ctx: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        # x_t, x_ctx: (B, T_samples) waveforms
        patches = patchify_waveform(x_t, self.cfg.patch_size)
        ctx = patchify_waveform(x_ctx, self.cfg.patch_size)
        h = self.in_proj(patches) + self.ctx_proj(ctx) + self.t_embed(t.view(-1, 1, 1))
        h = self.dit(h)
        pred_patches = self.out_proj(h)
        pred = unpatchify_waveform(pred_patches)
        if mask is not None and mask.dim() == 2:
            return pred * mask + x_ctx * (1.0 - mask)
        return pred
