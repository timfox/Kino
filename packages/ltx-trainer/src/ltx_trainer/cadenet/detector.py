"""YOLO detector stubs for Thread S (fast) and Thread Q (quality)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.cadenet.schema import Detection


class YOLOStub(nn.Module):
    """Grid-based detection stub; ``fast=True`` uses coarser stride."""

    def __init__(self, *, fast: bool = False, conf_thresh: float = 0.25) -> None:
        super().__init__()
        stride = 32 if fast else 16
        self.stride = stride
        self.conf_thresh = conf_thresh
        ch = 16 if fast else 32
        self.backbone = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, 4, 1),
        )

    @torch.no_grad()
    def detect(self, frame: Tensor, *, img_size: tuple[int, int] | None = None) -> list[Detection]:
        self.eval()
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        h, w = frame.shape[-2:]
        logits = self.backbone(frame)
        _, _, gh, gw = logits.shape
        out: list[Detection] = []
        for gy in range(gh):
            for gx in range(gw):
                conf = torch.sigmoid(logits[0, 0, gy, gx]).item()
                if conf < self.conf_thresh:
                    continue
                cx = (gx + 0.5) * w / gw
                cy = (gy + 0.5) * h / gh
                bw = w / gw * 1.5
                bh = h / gh * 1.5
                out.append(Detection(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, conf))
        return out
