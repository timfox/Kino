"""ViTMatte-style training losses (Eq. 14)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CineMatteLossConfig:
    unknown_width: int = 25
    laplacian_levels: int = 5


def trimap_from_alpha(alpha: Tensor, unknown_width: int = 25) -> Tensor:
    """Build fg=1, unknown=0.5, bg=0 trimap from ground-truth alpha."""
    fg = (alpha >= 0.95).float()
    bg = (alpha <= 0.05).float()
    unknown = 1.0 - fg - bg
    # erode fg/bg boundaries to widen unknown band
    if unknown_width > 0:
        k = unknown_width * 2 + 1
        fg_e = F.max_pool2d(fg, k, stride=1, padding=unknown_width)
        bg_e = F.max_pool2d(bg, k, stride=1, padding=unknown_width)
        unknown = (1.0 - fg_e - bg_e).clamp(min=0.0)
        fg = fg * (1.0 - unknown)
        bg = bg * (1.0 - unknown)
    return fg + 0.5 * unknown


def separate_l1_loss(pred: Tensor, target: Tensor, trimap: Tensor) -> Tensor:
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if trimap.dim() == 3:
        trimap = trimap.unsqueeze(1)
    fg = trimap >= 0.99
    bg = trimap <= 0.01
    unk = ~(fg | bg)
    loss = pred.new_tensor(0.0)
    n = 0
    if fg.any():
        loss = loss + F.l1_loss(pred[fg], target[fg])
        n += 1
    if bg.any():
        loss = loss + F.l1_loss(pred[bg], target[bg])
        n += 1
    if unk.any():
        loss = loss + F.l1_loss(pred[unk], target[unk])
        n += 1
    return loss / max(n, 1)


def _laplacian_pyramid(x: Tensor, levels: int) -> list[Tensor]:
    pyr: list[Tensor] = [x]
    cur = x
    for _ in range(levels - 1):
        cur = F.avg_pool2d(cur, 2)
        pyr.append(cur)
    return pyr


def laplacian_loss(pred: Tensor, target: Tensor, levels: int = 5) -> Tensor:
    pp = _laplacian_pyramid(pred, levels)
    pt = _laplacian_pyramid(target, levels)
    return sum(F.l1_loss(a, b) for a, b in zip(pp, pt)) / levels


def gradient_penalty_loss(pred: Tensor, target: Tensor) -> Tensor:
    def grads(t: Tensor) -> tuple[Tensor, Tensor]:
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        return gx, gy

    pgx, pgy = grads(pred)
    tgx, tgy = grads(target)
    return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


class CineMatteLoss:
    """``L_total = L_separate_l1 + L_lap + L_gp``."""

    def __init__(self, cfg: CineMatteLossConfig | None = None) -> None:
        self.cfg = cfg or CineMatteLossConfig()

    def __call__(self, pred: Tensor, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        trimap = trimap_from_alpha(target, self.cfg.unknown_width)
        l_sep = separate_l1_loss(pred, target, trimap)
        l_lap = laplacian_loss(pred, target, self.cfg.laplacian_levels)
        l_gp = gradient_penalty_loss(pred, target)
        total = l_sep + l_lap + l_gp
        stats = {
            "loss_total": float(total.detach()),
            "loss_sep_l1": float(l_sep.detach()),
            "loss_lap": float(l_lap.detach()),
            "loss_gp": float(l_gp.detach()),
        }
        return total, stats
