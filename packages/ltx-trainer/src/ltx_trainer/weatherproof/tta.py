"""Test-time augmentation (Sec. 1, Table 1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.weatherproof.model import UniMatchV2Seg


@torch.no_grad()
def predict_with_tta(model: UniMatchV2Seg, img: Tensor) -> Tensor:
    """Average softmax over flips and scales."""
    model.eval()
    views: list[Tensor] = [img]
    views.append(torch.flip(img, dims=[-1]))
    views.append(torch.flip(img, dims=[-2]))
    h, w = img.shape[-2:]
    small = F.interpolate(img.unsqueeze(0), scale_factor=0.875, mode="bilinear", align_corners=False)
    small = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    views.append(small)
    acc = None
    for v in views:
        logits = model.student(v.unsqueeze(0) if v.dim() == 3 else v)
        prob = F.softmax(logits, dim=1)
        acc = prob if acc is None else acc + prob
    assert acc is not None
    return acc.argmax(dim=1).squeeze(0)
