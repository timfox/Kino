"""GFCIP geodesic flow consistency (Sec. 3.2, Eq. 3–7)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def erp_to_spherical_daovi(x: Tensor, y: Tensor, width: int, height: int) -> tuple[Tensor, Tensor]:
    """Eq. (5–6): longitude φ and colatitude θ from ERP pixel indices."""
    phi = 2.0 * math.pi * (x.float() + 0.5) / width - math.pi
    theta = math.pi * (y.float() + 0.5) / height
    return phi, theta


def geodesic_distance_rad(phi_a: Tensor, theta_a: Tensor, phi_b: Tensor, theta_b: Tensor) -> Tensor:
    """Eq. (4): great-circle distance on unit sphere."""
    inner = (
        torch.cos(theta_a) * torch.cos(theta_b) * torch.cos(phi_a - phi_b)
        + torch.sin(theta_a) * torch.sin(theta_b)
    ).clamp(-1.0, 1.0)
    return torch.acos(inner)


def geodesic_distance_pixels(
    p: tuple[float, float],
    p_prime: tuple[float, float],
    width: int,
    height: int,
) -> float:
    """Scalar geodesic distance between ERP pixel locations."""
    phi_a, theta_a = erp_to_spherical_daovi(torch.tensor(p[0]), torch.tensor(p[1]), width, height)
    phi_b, theta_b = erp_to_spherical_daovi(torch.tensor(p_prime[0]), torch.tensor(p_prime[1]), width, height)
    return float(geodesic_distance_rad(phi_a, theta_a, phi_b, theta_b))


def flow_consistency_error(
    flow_fwd: Tensor,
    flow_bwd: Tensor,
    eps_deg: float = 0.4,
) -> tuple[Tensor, Tensor]:
    """
    Bidirectional flow consistency mask Mr (Eq. 2–3, 7).

    flow_fwd, flow_bwd: [B, 2, H, W] in pixel units.
    Returns validity [B, 1, H, W] and geodesic error map [B, 1, H, W] (radians).
    """
    b, _, h, w = flow_fwd.shape
    device = flow_fwd.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    phi0, theta0 = erp_to_spherical_daovi(xx, yy, w, h)
    # p' = p + fwd(p) + bwd(p + fwd(p))
    px = xx + flow_fwd[:, 0]
    py = yy + flow_fwd[:, 1]
    px2 = px + flow_bwd[:, 0]
    py2 = py + flow_bwd[:, 1]

    phi1, theta1 = erp_to_spherical_daovi(px2.reshape(-1), py2.reshape(-1), w, h)
    phi0f = phi0.reshape(1, -1).expand(b, -1)
    theta0f = theta0.reshape(1, -1).expand(b, -1)
    err = geodesic_distance_rad(phi0f, theta0f, phi1, theta1).view(b, 1, h, w)

    eps_rad = math.radians(eps_deg)
    valid = (err < eps_rad).float()
    return valid, err
