"""LALE end-to-end segmentation model (§3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lale.blocks import ConvMixerBlock, TransformerBlock
from ltx_trainer.lale.config import LaleConfig
from ltx_trainer.lale.ops import RMSNorm, StarReLU


class ConvolutionStem(nn.Module):
    """Two 3×3 stride-2 convolutions → H/4 (§3.1)."""

    def __init__(self, cfg: LaleConfig) -> None:
        super().__init__()
        c = cfg.stem_channels
        self.conv1 = nn.Conv2d(3, c, cfg.stem_kernel, stride=cfg.stem_stride, padding=1)
        self.norm1 = RMSNorm(c)
        self.act1 = StarReLU()
        self.conv2 = nn.Conv2d(c, c, cfg.stem_kernel, stride=cfg.stem_stride, padding=1)
        self.norm2 = RMSNorm(c)
        self.act2 = StarReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        return self.act2(self.norm2(self.conv2(x)))


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        use_transformer: bool,
        depth: int = 2,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            RMSNorm(out_ch),
        )
        block_cls = TransformerBlock if use_transformer else ConvMixerBlock
        self.blocks = nn.Sequential(*[block_cls(out_ch) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.down(x))


class MultiScaleDecoder(nn.Module):
    """All-MLP multi-scale decoder (§3.4)."""

    def __init__(self, cfg: LaleConfig) -> None:
        super().__init__()
        cdec = cfg.decoder_channels
        self.projs = nn.ModuleList(
            [nn.Conv2d(ch, cdec, 1) for ch in cfg.stage_channels]
        )
        self.fuse_norm = nn.BatchNorm2d(cdec * len(cfg.stage_channels))
        self.fuse_act = StarReLU()
        self.drop = nn.Dropout2d(cfg.decoder_dropout)
        self.head = nn.Conv2d(cdec * len(cfg.stage_channels), cfg.num_classes, 1)

    def forward(self, features: list[Tensor]) -> Tensor:
        target_hw = features[0].shape[-2:]
        aligned = []
        for proj, feat in zip(self.projs, features):
            y = proj(feat)
            if y.shape[-2:] != target_hw:
                y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
            aligned.append(y)
        fused = torch.cat(aligned, dim=1)
        fused = self.fuse_act(self.fuse_norm(fused))
        fused = self.drop(fused)
        return self.head(fused)


class LALE(nn.Module):
    """Resolution-bifurcated hybrid encoder + lightweight decoder."""

    def __init__(self, cfg: LaleConfig | None = None, scale: str = "S1") -> None:
        super().__init__()
        self.cfg = cfg or LaleConfig()
        self.scale = scale
        depth_mult = {"S1": 1, "S2": 2, "S3": 2, "S4": 3}.get(scale, 1)
        ch = self.cfg.stage_channels
        self.stem = ConvolutionStem(self.cfg)
        stages = []
        in_ch = ch[0]
        for i, out_ch in enumerate(ch):
            stage_id = i + 1
            use_tf = stage_id in self.cfg.transformer_stages
            stages.append(
                EncoderStage(
                    in_ch,
                    out_ch,
                    use_transformer=use_tf,
                    depth=depth_mult if use_tf else depth_mult + 1,
                    downsample=(i > 0),  # F1 stays at H/4 per §3.1
                )
            )
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.decoder = MultiScaleDecoder(self.cfg)

    def forward(self, x: Tensor) -> Tensor:
        f0 = self.stem(x)
        feats = [f0]
        h = f0
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        # feats[0]=stem, feats[1..4]=stages — decoder uses F1..F4 at indices 1-4
        logits = self.decoder(feats[1:])
        return F.interpolate(
            logits,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )


def count_parameters_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
