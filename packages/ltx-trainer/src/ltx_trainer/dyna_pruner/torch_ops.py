"""Torch implementations of Dyna-Pruner masks and synergy (arXiv:2606.15346)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.dyna_pruner.config import DynaPrunerConfig


def importance_from_temporal_variance(x: torch.Tensor) -> torch.Tensor:
    """
    Per-pixel importance from temporal (+ channel) variance.

    Accepts ``[B,C,F,H,W]``, ``[C,F,H,W]``, or ``[B,C,H,W]`` (single frame).
    Returns ``[H,W]`` or ``[B,H,W]``.
    """
    t = x
    batched = t.ndim == 5
    if t.ndim == 4:
        t = t.unsqueeze(0)
    if t.ndim != 5:
        raise ValueError(f"expected 4D/5D video/latent, got {tuple(t.shape)}")
    var = t.var(dim=(1, 2), unbiased=False)
    vmax = var.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    s = (var / vmax).clamp(0.0, 1.0)
    return s if batched else s[0]


def soft_data_mask(s: torch.Tensor) -> torch.Tensor:
    return s.clamp(0.0, 1.0)


def hard_mask_ste(s: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    soft = soft_data_mask(s)
    hard = (soft >= threshold).to(soft.dtype)
    return hard + soft - soft.detach()


def apply_data_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply ``[H,W]`` or ``[B,H,W]`` mask to video ``[B,C,F,H,W]``."""
    if x.ndim == 4:
        x = x.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and x.shape[0] > 1:
        mask = mask.expand(x.shape[0], -1, -1)
    m = mask.unsqueeze(1).unsqueeze(2)
    return x * m


def sparsity_fraction(mask: torch.Tensor) -> float:
    return float(1.0 - mask.mean().item())


def l1_sparsity_penalty(s: torch.Tensor) -> float:
    return float(s.abs().sum().item())


def data_threshold_for_sparsity(s: torch.Tensor, sd: float) -> float:
    flat = s.reshape(-1).sort().values
    idx = min(flat.numel() - 1, max(0, int(round(sd * flat.numel()))))
    return float(flat[idx].item()) if flat.numel() else 0.5


def aggregate_receptive_field(s: torch.Tensor, *, kernel: int = 3) -> torch.Tensor:
    h, w = s.shape[-2], s.shape[-1]
    k = max(1, kernel)
    bh = max(1, h // k)
    bw = max(1, w // k)
    blocks: list[torch.Tensor] = []
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            block = s[..., i : min(i + bh, h), j : min(j + bw, w)]
            blocks.append(block.mean())
    return torch.stack(blocks)


def model_mask_from_importance(importances: torch.Tensor, *, sw: float) -> torch.Tensor:
    k = importances.numel()
    if k == 0:
        return importances
    keep = max(1, int(round((1.0 - sw) * k)))
    order = torch.argsort(importances, descending=True)
    mask = torch.zeros_like(importances)
    mask[order[:keep]] = 1.0
    return mask


def block_skip_list_from_importance(
    s: torch.Tensor,
    *,
    num_blocks: int,
    sw: float,
) -> list[int]:
    """Map spatial importance to DiT block indices to skip (lowest scores pruned)."""
    if num_blocks <= 0:
        return []
    flat = s.reshape(-1)
    if flat.numel() < num_blocks:
        flat = F.pad(flat, (0, num_blocks - flat.numel()), value=flat.min().item())
    chunk = flat.numel() // num_blocks
    scores = torch.stack([flat[i * chunk : (i + 1) * chunk].mean() for i in range(num_blocks)])
    keep = max(1, int(round((1.0 - sw) * num_blocks)))
    order = torch.argsort(scores, descending=True)
    active = set(order[:keep].tolist())
    return [i for i in range(num_blocks) if i not in active]


def synchronized_masks(
    s: torch.Tensor,
    *,
    sd: float = 0.7,
    sw: float = 0.7,
    ste_threshold: float = 0.5,
    kernel: int = 3,
    num_blocks: int = 48,
) -> dict[str, Any]:
    if s.ndim == 3:
        s = s[0]
    soft = soft_data_mask(s)
    thresh = data_threshold_for_sparsity(soft, sd)
    m_data = hard_mask_ste(soft, threshold=thresh)
    importances = aggregate_receptive_field(soft, kernel=kernel)
    m_weight = model_mask_from_importance(importances, sw=sw)
    skip_blocks = block_skip_list_from_importance(soft, num_blocks=num_blocks, sw=sw)
    tau_val = 0.0
    if importances.numel():
        keep_i = max(0, min(importances.numel() - 1, int(round((1.0 - sw) * importances.numel())) - 1))
        tau_val = float(importances.sort(descending=True).values[keep_i].item())
    return {
        "S": soft,
        "M_data": m_data,
        "M_weight": m_weight,
        "I": importances,
        "tau": tau_val,
        "data_threshold": thresh,
        "skip_blocks": skip_blocks,
        "data_sparsity": sparsity_fraction(m_data),
        "model_keep_ratio": float(m_weight.mean().item()),
    }


def co_prune_video_batch(
    video: torch.Tensor,
    cfg: DynaPrunerConfig | None = None,
    *,
    num_blocks: int = 48,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Apply data mask per batch item; return masked video + serializable metadata."""
    cfg = cfg or DynaPrunerConfig()
    if video.ndim == 4:
        video = video.unsqueeze(0)
    layout = "bcfhw"
    if video.ndim == 5 and video.shape[1] != 3 and video.shape[2] == 3:
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        layout = "bfchw"
    out_items: list[dict[str, Any]] = []
    masked = video.clone()
    for b in range(video.shape[0]):
        sync = synchronized_masks(
            importance_from_temporal_variance(video[b : b + 1]),
            sd=cfg.data_sparsity_sd,
            sw=cfg.model_sparsity_sw,
            ste_threshold=cfg.ste_threshold,
            kernel=cfg.receptive_field,
            num_blocks=num_blocks,
        )
        masked[b] = apply_data_mask(video[b : b + 1], sync["M_data"])[0]
        out_items.append(
            {
                "data_sparsity": sync["data_sparsity"],
                "model_keep_ratio": sync["model_keep_ratio"],
                "data_threshold": sync["data_threshold"],
                "skip_blocks": sync["skip_blocks"],
                "tau": sync["tau"],
            }
        )
    if layout == "bfchw":
        masked = masked.permute(0, 2, 1, 3, 4).contiguous()
    return masked, out_items
