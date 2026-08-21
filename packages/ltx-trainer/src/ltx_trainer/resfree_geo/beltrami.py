"""Quasi-conformal / Beltrami theory (Sec. 3.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def beltrami_from_uv(u: Tensor, v: Tensor) -> Tensor:
    """Finite-difference Beltrami coefficient μ from displacement (u, v)."""
    du_dy, du_dx = torch.gradient(u)
    dv_dy, dv_dx = torch.gradient(v)
    fz_re = 0.5 * (du_dx + dv_dy)
    fz_im = -0.5 * (du_dy - dv_dx)
    fzb_re = 0.5 * (du_dx - dv_dy)
    fzb_im = 0.5 * (du_dy + dv_dx)
    denom = (fz_re**2 + fz_im**2).clamp(min=1e-8)
    mu_re = (fzb_re * fz_re + fzb_im * fz_im) / denom
    mu_im = (fzb_im * fz_re - fzb_re * fz_im) / denom
    return torch.sqrt(mu_re**2 + mu_im**2)


def mu_magnitude(mu: Tensor) -> Tensor:
    return mu.abs() if mu.is_complex() else mu.abs()


def maximal_dilatation(mu: Tensor, eps: float = 1e-6) -> Tensor:
    """K = (1 + |μ|) / (1 - |μ|) (Eq. 2)."""
    m = mu_magnitude(mu).clamp(max=1.0 - eps)
    return (1.0 + m) / (1.0 - m)


def beltrami_energy(mu: Tensor) -> Tensor:
    """E_B = ∫ |μ|² (Eq. 3 discretized)."""
    m = mu_magnitude(mu)
    return (m**2).mean()


def synthesize_beltrami_field(h: int, w: int, *, n_modes: int = 5, k_max: float = 0.9) -> Tensor:
    """Eq. 22: randomized sinusoidal μ with ‖μ‖∞ < 1."""
    xs = torch.linspace(0, 1, w)
    ys = torch.linspace(0, 1, h)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    mu = torch.zeros(h, w)
    for _ in range(n_modes):
        a = torch.empty(1).uniform_(-0.5, 0.5)
        fx = 10 ** torch.empty(1).uniform_(-1, 0.9)
        fy = 10 ** torch.empty(1).uniform_(-1, 0.9)
        mu = mu + a * torch.sin(2 * torch.pi * fx * xx) * torch.cos(2 * torch.pi * fy * yy)
    mu = mu / (mu.abs().max() + 1e-6) * k_max
    mu[0, :] = mu[-1, :] = mu[:, 0] = mu[:, -1] = 0.0
    return mu


def mild_sinusoidal_map(a: float = 0.08, b: float = 0.08, n: int = 32) -> tuple[Tensor, Tensor]:
    """Eq. 23 test deformation."""
    xs = torch.linspace(0, 1, n)
    ys = torch.linspace(0, 1, n)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    u = xx + a * torch.sin(torch.pi * xx) * torch.sin(2 * torch.pi * yy)
    v = yy + b * torch.sin(2 * torch.pi * xx) * torch.sin(torch.pi * yy)
    return u, v
