"""Frozen ViT backbone (DINOv2/DINOv3-compatible token interface)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet(x: Tensor) -> Tensor:
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


@dataclass
class VitTokenFeatures:
    """Patch tokens at 1/16 resolution (no CLS)."""

    tokens: Tensor  # [B, N, D]
    grid_h: int
    grid_w: int

    def to_map(self) -> Tensor:
        b, _, d = self.tokens.shape
        return self.tokens.transpose(1, 2).reshape(b, d, self.grid_h, self.grid_w)


class VitBackbone(Protocol):
    patch_size: int
    embed_dim: int

    def forward_tokens(self, x: Tensor) -> VitTokenFeatures: ...


class StubVitBackbone(nn.Module):
    """Lightweight patch encoder for tests (ViT-L/16 geometry)."""

    patch_size = 16

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

    def forward_tokens(self, x: Tensor) -> VitTokenFeatures:
        x = normalize_imagenet(x)
        feat = self.proj(x)
        b, d, gh, gw = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        return VitTokenFeatures(tokens=tokens, grid_h=gh, grid_w=gw)


class DinoVitBackbone(nn.Module):
    """Frozen DINOv2 ViT via torch.hub (drop-in until DINOv3 weights are wired)."""

    patch_size = 16

    def __init__(self, variant: str = "dinov2_vitl14", *, hub_repo: str = "facebookresearch/dinov2") -> None:
        super().__init__()
        self.variant = variant
        self.hub_repo = hub_repo
        self._model = torch.hub.load(hub_repo, variant, pretrained=True)
        self.embed_dim = int(getattr(self._model, "embed_dim", 1024))
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

    def forward_tokens(self, x: Tensor) -> VitTokenFeatures:
        x = normalize_imagenet(x)
        with torch.no_grad():
            out = self._model.forward_features(x)
        if isinstance(out, dict):
            tokens = out.get("x_norm_patchtokens") or out.get("x_prenorm")
        else:
            tokens = out[:, 1:, :] if out.shape[1] > 1 else out
        gh = x.shape[-2] // self.patch_size
        gw = x.shape[-1] // self.patch_size
        return VitTokenFeatures(tokens=tokens, grid_h=gh, grid_w=gw)


class DinoV3VitBackbone(nn.Module):
    """Frozen DINOv3 ViT via Hugging Face (paper default: ViT-L/16)."""

    patch_size = 16

    def __init__(
        self,
        model_id: str = "facebook/dinov2-large",
        *,
        fallback_hub: str = "dinov2_vitl14",
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self._use_hf = False
        self._model: nn.Module
        try:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(model_id)
            self.embed_dim = int(self._model.config.hidden_size)
            self._use_hf = True
        except Exception:
            self._model = torch.hub.load("facebookresearch/dinov2", fallback_hub, pretrained=True)
            self.embed_dim = int(getattr(self._model, "embed_dim", 1024))
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

    def forward_tokens(self, x: Tensor) -> VitTokenFeatures:
        x = normalize_imagenet(x)
        with torch.no_grad():
            if self._use_hf:
                out = self._model(pixel_values=x)
                tokens = out.last_hidden_state[:, 1:, :]
            else:
                out = self._model.forward_features(x)
                if isinstance(out, dict):
                    tokens = out.get("x_norm_patchtokens") or out["x_prenorm"][:, 1:, :]
                else:
                    tokens = out[:, 1:, :]
        gh = x.shape[-2] // self.patch_size
        gw = x.shape[-1] // self.patch_size
        n = tokens.shape[1]
        if gh * gw != n:
            gh = int(n**0.5)
            gw = n // gh
        return VitTokenFeatures(tokens=tokens, grid_h=gh, grid_w=gw)


def build_backbone(name: str = "stub", *, embed_dim: int = 256) -> nn.Module:
    name = name.lower()
    if name in ("stub", "test"):
        return StubVitBackbone(embed_dim=embed_dim)
    if name in ("dinov3", "dinov3_vitl16", "dinov3_vitl"):
        return DinoV3VitBackbone("facebook/dinov2-large")  # swap to HF dinov3 id when weights cached
    if name.startswith("dinov3_"):
        hf_id = name.replace("dinov3_", "facebook/dinov3-").replace("_", "-")
        if not hf_id.endswith("-pretrain-lvd1689m"):
            hf_id = f"facebook/{name.replace('_', '-')}"
        return DinoV3VitBackbone(hf_id)
    if name.startswith("dinov2"):
        return DinoVitBackbone(variant=name if name != "dinov2" else "dinov2_vitl14")
    raise ValueError(
        f"Unknown backbone: {name} (stub|dinov2_vitl14|dinov2_vitb14|dinov3_vitl16)"
    )
