"""PAR MAR transformer + MLP denoiser stub (Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.par.circular_padding import cyclic_translate_width
from ltx_trainer.par.config import PARConfig
from ltx_trainer.par.losses import consistency_loss, total_loss, vanilla_mar_loss
from ltx_trainer.par.masking import random_mask


class PARStub(nn.Module):
    """Frozen-VAE surrogate: patch tokens → transformer f → MLP ε_θ."""

    def __init__(self, cfg: PARConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PARConfig()
        d = self.cfg.latent_dim
        self.patch_embed = nn.Linear(3, d)
        self.text_proj = nn.Linear(self.cfg.text_dim, d)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=d * 4, batch_first=True),
            num_layers=2,
        )
        self.cond_proj = nn.Linear(d, d)
        self.mlp_denoise = nn.Sequential(
            nn.Linear(d + 1, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.rgb_head = nn.Linear(d, 3)

    def encode_patches(self, rgb: Tensor) -> Tensor:
        """Patchify RGB → latent tokens (VAE encoder surrogate)."""
        b, c, h, w = rgb.shape
        p = int(self.cfg.num_patches**0.5)
        ph, pw = h // p, w // p
        patches = rgb.unfold(2, ph, ph).unfold(3, pw, pw)
        patches = patches.contiguous().view(b, c, -1, ph, pw).mean(dim=(-1, -2))
        patches = patches.permute(0, 2, 1)
        return self.patch_embed(patches)

    def forward_mar(
        self,
        tokens: Tensor,
        text: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict noise on masked positions."""
        b, n, d = tokens.shape
        m = mask.view(b, n, 1)
        masked = tokens * (1 - m) + self.cond_proj(text).unsqueeze(1) * m
        z = self.encoder(masked + self.text_proj(text).unsqueeze(1))
        t = torch.rand(b, 1, 1, device=tokens.device)
        xt = tokens + t * torch.randn_like(tokens)
        pred = self.mlp_denoise(torch.cat([xt, t.expand(b, n, 1)], dim=-1))
        return pred, z

    def forward(
        self,
        rgb: Tensor,
        text: Tensor,
        *,
        shift: int | None = None,
    ) -> dict[str, Tensor]:
        tokens = self.encode_patches(rgb)
        mask = random_mask(tokens.shape[1], self.cfg.mask_ratio).to(rgb.device)
        mask = mask.view(1, -1).expand(tokens.shape[0], -1)
        noise = torch.randn_like(tokens)
        pred, _ = self.forward_mar(tokens, text, mask)
        l_va = vanilla_mar_loss(noise, pred, mask)

        shift = shift if shift is not None else max(1, self.cfg.width // 16)
        rgb_s = cyclic_translate_width(rgb, shift)
        tokens_s = self.encode_patches(rgb_s)
        mask_s = torch.roll(mask, shifts=shift, dims=-1)
        pred_s, _ = self.forward_mar(tokens_s, text, mask_s)
        pred_aligned = torch.roll(pred, shifts=shift, dims=1)
        l_cons = consistency_loss(pred_aligned, pred_s, mask_s)

        loss = total_loss(l_va, l_cons, self.cfg)
        recon = self._decode_tokens(tokens - pred.detach() * mask.unsqueeze(-1), rgb.shape)
        return {
            "L": loss,
            "L_va": l_va,
            "L_consistency": l_cons,
            "pred_rgb": recon,
            "mask": mask,
        }

    def _decode_tokens(self, tokens: Tensor, shape: tuple[int, ...]) -> Tensor:
        b, _c, h, w = shape
        feat = tokens.mean(dim=1)
        rgb = torch.sigmoid(self.rgb_head(feat)).view(b, 3, 1, 1)
        return F.interpolate(rgb, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
