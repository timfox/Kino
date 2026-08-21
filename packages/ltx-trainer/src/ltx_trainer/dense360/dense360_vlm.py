"""Dense360VLM stub: ERP-RoPE indices + grounding head (Sec. 4.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.dense360.config import SEG_TOKEN, Dense360Config
from ltx_trainer.dense360.erp_rope import erp_position_grid, latitude_gamma, token_erp_coords


class Dense360VLMStub(nn.Module):
    """Vision-encoder-projector-LLM stub with ERP-RoPE coordinate injection."""

    def __init__(self, cfg: Dense360Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Dense360Config()
        d = self.cfg.feature_dim
        p = self.cfg.patch_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, kernel_size=p, stride=p),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(d + 2, d, kernel_size=1)
        self.seg_head = nn.Conv2d(d, 1, 1)
        self.caption_head = nn.Linear(d, d)

    def _visual_tokens(self, erp: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, _, h, w = erp.shape
        feat = self.encoder(erp)
        _, _, gh, gw = feat.shape
        if self.cfg.use_erp_rope:
            th, tw = token_erp_coords(gh, gw, self.cfg.patch_size, h, w, device=erp.device)
            th = th.view(1, gh, gw).expand(b, -1, -1)
            tw = tw.view(1, gh, gw).expand(b, -1, -1)
        else:
            th = torch.arange(gh, device=erp.device).view(1, gh, 1).expand(b, gh, gw).float()
            tw = torch.arange(gw, device=erp.device).view(1, 1, gw).expand(b, gh, gw).float()
        pos = torch.stack([th, tw], dim=1)
        fused = self.proj(torch.cat([feat, pos], dim=1))
        return fused, th, tw

    def forward(
        self,
        erp: Tensor,
        *,
        referring_text: str | None = None,
    ) -> dict[str, Tensor]:
        """
        erp [B,3,H,W] → seg logits [B,1,H,W] and pooled text features.
        """
        fused, _, _ = self._visual_tokens(erp)
        logits = self.seg_head(fused)
        mask = F.interpolate(logits, size=erp.shape[-2:], mode="bilinear", align_corners=False)
        pooled = fused.mean(dim=(2, 3))
        caption_vec = self.caption_head(pooled)
        return {
            "seg_logits": mask,
            "caption_vec": caption_vec,
            "seg_token": torch.tensor([hash(SEG_TOKEN) % 1000], device=erp.device),
            "referring": referring_text or "",
        }

    def erp_rope_demo(self) -> dict[str, float]:
        h, w = self.cfg.height, self.cfg.width
        _, col_w = erp_position_grid(h, w)
        center_w = col_w[h // 2, w // 2].item()
        edge_w = col_w[h // 2, 0].item()
        return {
            "center_gt_edge": float(center_w > edge_w),
            "gamma_positive": float(latitude_gamma(h, w) > 0),
        }
