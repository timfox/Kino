"""Relightable appearance: diffuse + specular separation (Sec. 7.3–7.4)."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False


if _TORCH:

    class GeoLiftingStub(nn.Module):
        """Temporal normal stack -> albedo + normal (Eq. 11)."""

        def __init__(self, ch: int = 9) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ch, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 6, 1),
            )

        def forward(self, normal_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.net(normal_stack)
            return out[:, :3].sigmoid(), out[:, 3:6].tanh() * 0.5 + 0.5

    class DiffuseNetStub(nn.Module):
        """Eq. 20: (a, rho, n, chi) -> Gaussian diffuse color."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(8, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 4, 1),
            )

        def forward(self, a: torch.Tensor, rho: torch.Tensor, n: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
            x = torch.cat([a, rho, n, chi], dim=1)
            return self.net(x).sigmoid()

    class SpecularNetStub(nn.Module):
        """Cross-attention style specular shading (Eq. 21–22), simplified."""

        def __init__(self, ray_dim: int = 5, hidden: int = 32) -> None:
            super().__init__()
            self.ray_proj = nn.Linear(ray_dim, hidden)
            self.base = nn.Sequential(
                nn.Conv2d(7, hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 1, 1),
            )

        def forward(
            self,
            a: torch.Tensor,
            rho: torch.Tensor,
            n: torch.Tensor,
            ray_enc: torch.Tensor,
        ) -> torch.Tensor:
            # ray_enc: B, R, H, W, D
            b, r, h, w, d = ray_enc.shape
            keys = self.ray_proj(ray_enc.reshape(b, r, h * w, d))
            attn = torch.softmax(keys.mean(dim=1), dim=-1)  # B, H*W, hidden
            attn = attn.reshape(b, h, w, -1).permute(0, 3, 1, 2)
            base = self.base(torch.cat([a, rho, n], dim=1))
            return torch.nn.functional.softplus(base + attn[:, :1])


def composite_shading(
    diffuse: np.ndarray,
    specular: np.ndarray,
) -> np.ndarray:
    """Final shading c_spec + c_diff (Sec. 7.5)."""
    return np.clip(diffuse + specular, 0.0, None).astype(np.float32)
