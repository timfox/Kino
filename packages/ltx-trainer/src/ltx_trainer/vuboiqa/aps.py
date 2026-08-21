"""Adaptive prior-equator sampling (Sec. 3.2, Eq. 1–2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.vuboiqa.config import VuBoiqaConfig


def pe_cdf(x_deg: float, mu: float, lambda_: float) -> float:
    """Logistic CDF for prior-equator latitude (unnormalized grid split)."""
    z = (x_deg - mu) / max(lambda_, 1e-6)
    return 1.0 / (1.0 + math.exp(-z))


def grid_split_k_low(k: int, theta_t: int, mu: float, lam: float) -> int:
    """Eq. 1: patches in low-latitude band [-θ_T, θ_T] (stub mass estimate)."""
    steps = 200
    lo, hi = -theta_t, theta_t
    mass = 0.0
    for i in range(steps):
        x = lo + (hi - lo) * (i + 0.5) / steps
        mass += pe_cdf(x, mu, lam) - pe_cdf(lo, mu, lam)
    mass = min(1.0, max(0.0, mass / steps))
    return max(1, min(k - 1, int(k * mass))) if k > 1 else 1


def sample_patch_centers(
    erp_h: int,
    erp_w: int,
    cfg: VuBoiqaConfig,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, int, int]:
    """
    Return K patch top-left (y, x) and patch height/width in ERP pixels.
    Longitude uniform; latitude biased to equator (stub via normal around mu).
    """
    k = cfg.num_patches
    k_low = grid_split_k_low(k, cfg.theta_t, cfg.pe_mu_deg, cfg.pe_lambda_deg)
    k_low = max(1, min(k - 1, k_low))
    ph = max(4, int(erp_h * cfg.kappa_h))
    pw = max(4, int(erp_w * cfg.kappa_w))
    ys: list[int] = []
    xs: list[int] = []
    for i in range(k):
        # equator-biased latitude
        lat = torch.randn((), generator=generator).item() * cfg.pe_lambda_deg + cfg.pe_mu_deg
        y = int((lat / 180.0 + 0.5) * erp_h) - ph // 2
        x = int(torch.rand((), generator=generator).item() * max(1, erp_w - pw))
        y = max(0, min(erp_h - ph, y))
        xs.append(x)
        ys.append(y)
    return torch.tensor(list(zip(ys, xs))), ph, pw


def crop_patches(erp: Tensor, cfg: VuBoiqaConfig) -> Tensor:
    """erp: B×3×H×W → B×K×3×patch×patch."""
    b, c, h, w = erp.shape
    centers, ph, pw = sample_patch_centers(h, w, cfg)
    k = centers.shape[0]
    patches: list[Tensor] = []
    for i in range(k):
        y0, x0 = int(centers[i, 0]), int(centers[i, 1])
        patch = erp[0, :, y0 : y0 + ph, x0 : x0 + pw]
        patch = torch.nn.functional.interpolate(
            patch.unsqueeze(0),
            size=(cfg.patch_size, cfg.patch_size),
            mode="bilinear",
            align_corners=False,
        )
        patches.append(patch)
    stacked = torch.cat(patches, dim=0).unsqueeze(0)  # 1×K×3×P×P
    return stacked.expand(b, -1, -1, -1, -1)
