"""Video Merging Model (VMM) — per-frame learned fusion of exposure brackets."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.hdr_ingest import merge_log_domain_radiance


class VideoMergingModel(nn.Module):
    """Lightweight per-pixel MLP + exposure self-attention (Tedla et al. Sec. 4)."""

    def __init__(
        self,
        *,
        num_exposures: int = 3,
        hidden: int = 128,
        embed: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_exposures = num_exposures
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed),
        )
        self.attn = nn.MultiheadAttention(embed, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, 1)

    def forward(self, brackets: Tensor, ev_list: list[float]) -> Tensor:
        if brackets.ndim != 5:
            raise ValueError(f"Expected [N,C,F,H,W], got {tuple(brackets.shape)}")
        n, c, f, h, w = brackets.shape
        if len(ev_list) != n:
            raise ValueError(f"len(ev_list)={len(ev_list)} != N={n}")
        if n != self.num_exposures:
            raise ValueError(f"Model expects {self.num_exposures} exposures, got {n}")

        gamma = 2.2
        ldr = brackets.clamp(0.0, 1.0)
        ev = torch.tensor(ev_list, device=brackets.device, dtype=brackets.dtype)
        linear = ldr.pow(gamma)
        radiance = torch.stack([linear[k] / (2.0**ev[k]) for k in range(n)], dim=0)

        feat7 = torch.stack(
            [
                torch.cat(
                    [
                        ldr[k],
                        radiance[k],
                        ldr.new_full((1, f, h, w), float(ev_list[k])),
                    ],
                    dim=0,
                )
                for k in range(n)
            ],
            dim=0,
        )
        pixels = feat7.permute(2, 3, 4, 0, 1).reshape(f * h * w, n, 7)
        z = self.mlp(pixels)
        z_n = self.norm(z)
        z = z + self.attn(z_n, z_n, z_n, need_weights=False)[0]
        weights = F.softmax(self.head(z).squeeze(-1), dim=-1)

        rad = radiance.permute(2, 3, 4, 0, 1).reshape(f * h * w, n, c)
        hdr_pix = (rad * weights.unsqueeze(-1)).sum(dim=1)
        return hdr_pix.reshape(f, h, w, c).permute(3, 0, 1, 2)


def merge_brackets_debevec(brackets: Tensor, ev_list: list[float], *, gamma: float = 2.2) -> Tensor:
    return merge_log_domain_radiance(brackets, ev_list, gamma=gamma)


def vmm_log_loss(pred_cfhw: Tensor, target_cfhw: Tensor, *, eps: float = 1e-6) -> Tensor:
    if pred_cfhw.ndim == 5:
        pred_cfhw = pred_cfhw.squeeze(0)
    if target_cfhw.ndim == 5:
        target_cfhw = target_cfhw.squeeze(0)
    peak = target_cfhw.reshape(-1).quantile(0.999).clamp(min=eps)
    pred_n = pred_cfhw / peak
    tgt_n = target_cfhw / peak
    return F.l1_loss(torch.log(pred_n + eps), torch.log(tgt_n + eps))


def load_vmm_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> VideoMergingModel:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = VideoMergingModel(
        num_exposures=int(cfg.get("num_exposures", 3)),
        hidden=int(cfg.get("hidden", 128)),
        embed=int(cfg.get("embed", 64)),
        num_heads=int(cfg.get("num_heads", 4)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.to(device)
