"""Benchmarked depth model registry (Table 1–2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.sphere_depth.benchmarks import table2_by_model
from ltx_trainer.sphere_depth.disparity import disparity_to_depth, relative_depth_stub


@dataclass
class DepthModelSpec:
    name: str
    full_360: bool
    uses_disparity: bool
    ref_lambda: float
    ref_gravity_epsilon: float


def model_specs() -> dict[str, DepthModelSpec]:
    t2 = table2_by_model()
    specs = {
        "ACDNet": DepthModelSpec("ACDNet", True, False, t2["ACDNet"]["lambda"], t2["ACDNet"]["gravity_aligned"]),
        "DepthAnywhere": DepthModelSpec(
            "DepthAnywhere", True, True, t2["DepthAnywhere"]["lambda"], t2["DepthAnywhere"]["gravity_aligned"]
        ),
        "BiFuse++": DepthModelSpec("BiFuse++", True, True, t2["BiFuse++"]["lambda"], t2["BiFuse++"]["gravity_aligned"]),
        "SliceNet": DepthModelSpec("SliceNet", True, False, t2["SliceNet"]["lambda"], t2["SliceNet"]["gravity_aligned"]),
        "DepthAnythingV2": DepthModelSpec("DepthAnythingV2", False, True, 1.38, 1.72),
    }
    return specs


class StubDepthEstimator(nn.Module):
    """Trainable-free depth stub keyed to paper λ prior."""

    def __init__(self, spec: DepthModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, erp: Tensor) -> Tensor:
        scale = self.spec.ref_lambda * (1.0 + 0.05 * torch.tanh(self.bias.detach()))
        if self.spec.uses_disparity:
            disp = torch.randn(erp.shape[0], 1, erp.shape[2], erp.shape[3], device=erp.device) * 0.1
            return disparity_to_depth(disp.squeeze(1))
        return relative_depth_stub(erp, scale=float(scale.item()))


def build_estimator(name: str) -> StubDepthEstimator:
    spec = model_specs()[name]
    return StubDepthEstimator(spec)


def predict_landmark_depths(
    depth_map: Tensor,
    u: Tensor,
    v: Tensor,
) -> Tensor:
    """Bilinear sample depth at landmark pixels (u, v)."""
    b, h, w = depth_map.shape[0], depth_map.shape[-2], depth_map.shape[-1]
    gx = (u / (w - 1)) * 2.0 - 1.0
    gy = (v / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(b, -1, 1, 2)
    sampled = nn.functional.grid_sample(
        depth_map.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(b, -1).squeeze(0)
