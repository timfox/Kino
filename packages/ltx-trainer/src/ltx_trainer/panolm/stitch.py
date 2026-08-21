"""Panorama synthesis stub (OneBEV-style ray casting, Fig. 3 / Supp. B.2.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def first_hit_panorama_stub(
    num_cameras: int = 6,
    height: int = 32,
    width: int = 64,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Return a synthetic ERP RGB panorama (B=1, 3, H, W)."""
    dev = device or torch.device("cpu")
    # Camera priority stripes mimic deterministic overlap resolution
    pano = torch.zeros(1, 3, height, width, device=dev)
    stripe = max(1, width // num_cameras)
    for i in range(num_cameras):
        pano[:, :, :, i * stripe : (i + 1) * stripe] = (i + 1) / num_cameras
    return pano
