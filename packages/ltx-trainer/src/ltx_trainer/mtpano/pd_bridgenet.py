"""Panorama-Dual-BridgeNet stub (Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.mtpano.config import MTPanoConfig
from ltx_trainer.mtpano.bridge import BridgeFeatureExtractor
from ltx_trainer.mtpano.erp import ERPTokenMixer, erp_phi_grid, film_condition_map


class FiLMModulation(nn.Module):
    """Geometry-aware FiLM for variant stream (Sec. 3.2.1)."""

    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, feat_dim * 2), nn.ReLU(), nn.Linear(feat_dim * 2, feat_dim * 2))

    def forward(self, feat: Tensor, cond: Tensor) -> Tensor:
        b, c, h, w = feat.shape
        params = self.mlp(cond.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        gamma, beta = params.chunk(2, dim=1)
        return (1 + gamma) * feat + beta


class TruncatedBridge(nn.Module):
    """Cross-stream bridge with detached opposing gradients (Sec. 3.2.2)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

    def forward(self, inv: Tensor, var: Tensor) -> tuple[Tensor, Tensor]:
        # inv, var: (B, HW, C)
        inv_q = inv + self.cross(inv, var.detach(), var.detach())[0]
        var_q = var + self.cross(var, inv.detach(), inv.detach())[0]
        return inv_q, var_q


class PDBridgeNetStub(nn.Module):
    def __init__(self, cfg: MTPanoConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MTPanoConfig()
        c = self.cfg.embed_dim
        self.backbone = nn.Conv2d(3, c, kernel_size=4, stride=4)
        self.mixer_inv = ERPTokenMixer(c)
        self.mixer_var = ERPTokenMixer(c)
        self.film = FiLMModulation(cond_dim=5, feat_dim=c)  # ray(3) + pos(2)
        self.bridge = TruncatedBridge(c)
        self.bfe_inv = BridgeFeatureExtractor(c)
        self.bfe_var = BridgeFeatureExtractor(c)
        self.head_sem = nn.Conv2d(c, 19, 1)
        self.head_depth = nn.Conv2d(c, 1, 1)
        self.head_norm = nn.Conv2d(c, 3, 1)
        self.head_grad = nn.Conv2d(c, 2, 1)
        self.head_edf = nn.Conv2d(c, 1, 1)
        self.head_point = nn.Conv2d(c, 3, 1)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        b, _, h, w = image.shape
        phi = erp_phi_grid(h, w, device=image.device)
        cond = film_condition_map(h, w, device=image.device, dtype=image.dtype).unsqueeze(0).expand(b, -1, -1, -1)

        f = self.backbone(image)
        c, fh, fw = f.shape[1], f.shape[2], f.shape[3]
        phi_f = F.interpolate(
            phi.unsqueeze(0).unsqueeze(0),
            size=f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        cond_f = F.interpolate(cond, size=f.shape[-2:], mode="bilinear", align_corners=False)
        inv = self.mixer_inv(f, phi_f)
        var = self.film(self.mixer_var(f, phi_f), cond_f)

        inv_flat = inv.flatten(2).transpose(1, 2)
        var_flat = var.flatten(2).transpose(1, 2)
        inv_b, var_b = self.bridge(inv_flat, var_flat)
        inv_b = self.bfe_inv(inv_b, var_b, detach_kv=True)
        var_b = self.bfe_var(var_b, inv_b, detach_kv=True)
        inv = inv_b.transpose(1, 2).reshape(b, c, fh, fw)
        var = var_b.transpose(1, 2).reshape(b, c, fh, fw)

        fused = inv + var
        up = lambda t: F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)

        return {
            "semseg": up(self.head_sem(fused)),
            "depth": up(self.head_depth(fused)).relu() + 1e-3,
            "normals": F.normalize(up(self.head_norm(fused)), dim=1),
            "grad": up(self.head_grad(fused)),
            "edf": up(self.head_edf(fused)).relu(),
            "point_map": up(self.head_point(fused)),
            "inv": up(inv),
            "var": up(var),
        }
