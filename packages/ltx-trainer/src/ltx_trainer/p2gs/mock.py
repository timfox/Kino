"""Runnable evaluation smoke for P2GS (arXiv:2605.16925)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ltx_trainer.p2gs.gaussians import GaussianModel
from ltx_trainer.p2gs.losses import P2GSLosses, build_hdr_pairs, hdr_inconsistency_score, std_luminance
from ltx_trainer.p2gs.photometric import ViewPhotometricParams, render_ldr, tone_map_gamma
from ltx_trainer.p2gs.rasterize import render_linear_hdr


def evaluation_smoke() -> dict[str, Any]:
    device = "cpu"
    vp = ViewPhotometricParams(2)
    vp._log_exposure.data = torch.tensor([0.0, 0.6931])
    hdr = torch.ones(3, 16, 16) * 0.5
    ldr = render_ldr(hdr, exposure=1.0, gamma=2.2)
    pts = torch.tensor([[0.0, 0.0, 5.0], [0.3, 0.0, 5.0]])
    g = GaussianModel.from_point_cloud(pts, scale=0.1).to(device)
    from ltx_trainer.p2gs.cameras import Camera

    cam = Camera(
        Path("dummy.png"),
        32,
        32,
        40.0,
        40.0,
        16.0,
        16.0,
        torch.eye(3),
        torch.zeros(3),
    )
    rendered = render_linear_hdr(g, cam, max_gaussians=4)
    pairs = build_hdr_pairs({0: hdr, 1: hdr * 2}, vp, [(0, 1)])
    loss_fn = P2GSLosses()
    l_exp = loss_fn.relative_exposure(pairs[0][0], pairs[0][1], alpha_ij=pairs[0][2])
    seq = torch.stack([ldr, ldr * 0.9])
    return {
        "package": "p2gs",
        "paper": "arXiv:2605.16925",
        "ldr_mean": round(float(ldr.mean()), 4),
        "tone_map": round(float(tone_map_gamma(hdr, 2.2).mean()), 4),
        "render_shape": list(rendered.shape),
        "l_exp": round(float(l_exp), 6),
        "his": round(hdr_inconsistency_score(seq, torch.ones(2)), 4),
        "std_lum": round(std_luminance(seq.unsqueeze(0)), 4),
    }
