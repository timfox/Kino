"""End-to-end VDP-HDR module (bracket prior + Fusion UNet)."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.vdp_hdr.bracket import BracketConfig, ldr_bracket_from_single
from ltx_trainer.vdp_hdr.fusion import FusionUNet, bracket_to_linear, fuse_bracket, mertens_fusion


@dataclass
class VdpHdrConfig:
    num_frames: int = 5
    gamma: float = 2.2
    ev_span: float = 5.0
    use_fusion_unet: bool = True
    fusion_backend: str = "unet"  # unet | mertens


class VdpHdr(nn.Module):
    """Single-shot HDR: synthesize LDR bracket, fuse to linear HDR (relative scale)."""

    def __init__(self, config: VdpHdrConfig | None = None) -> None:
        super().__init__()
        self.config = config or VdpHdrConfig()
        self.bracket_cfg = BracketConfig(
            num_frames=self.config.num_frames,
            gamma=self.config.gamma,
            ev_span=self.config.ev_span,
        )
        self.fusion: FusionUNet | None = None
        if self.config.use_fusion_unet:
            self.fusion = FusionUNet(self.config.num_frames)

    def predict_bracket(self, ldr_chw: Tensor) -> Tensor:
        """``[1,N,C,H,W]`` γ bracket from conditioning LDR."""
        return ldr_bracket_from_single(ldr_chw, self.bracket_cfg)

    def fuse(self, bracket_bncHW: Tensor) -> Tensor:
        """Fuse γ bracket → linear HDR ``[B,C,H,W]``."""
        lin = bracket_to_linear(bracket_bncHW, gamma=self.config.gamma)
        if self.config.fusion_backend == "mertens" or self.fusion is None:
            return mertens_fusion(lin)
        w = self.fusion(lin)
        return fuse_bracket(lin, w)

    def forward(self, ldr_chw: Tensor) -> tuple[Tensor, Tensor]:
        """``ldr [C,H,W]`` or ``[B,C,H,W]`` → ``(hdr, bracket)``."""
        batched = ldr_chw.ndim == 4
        if not batched:
            ldr_chw = ldr_chw.unsqueeze(0)
        brackets = []
        for i in range(ldr_chw.shape[0]):
            brackets.append(self.predict_bracket(ldr_chw[i]))
        bracket = torch.cat(brackets, dim=0)
        hdr = self.fuse(bracket)
        if not batched:
            hdr = hdr.squeeze(0)
            bracket = bracket.squeeze(0)
        return hdr, bracket
