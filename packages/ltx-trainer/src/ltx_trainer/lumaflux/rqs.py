"""Rational-Quadratic Spline tone-field decoder (Sec. 4.6, Eq. 16–17)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RQSToneField(nn.Module):
    """
    Monotone rational-quadratic spline expansion for luma Y.

    Lightweight parametric head; uses piecewise-linear monotone approximation
    when full spline params are predicted from context.
    """

    def __init__(self, num_knots: int = 8, hidden: int = 32) -> None:
        super().__init__()
        self.num_knots = num_knots
        self.param_head = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, num_knots * 3),
        )
        self.chroma_refine = nn.Conv2d(2, 2, 1)

    def _monotone_map(self, y: Tensor, logits: Tensor) -> Tensor:
        """Apply learned monotone knot slopes to expand dynamic range."""
        b = y.shape[0]
        params = logits.view(b, self.num_knots, 3)
        widths = F.softmax(params[:, :, 0], dim=-1)
        heights = F.softmax(params[:, :, 1], dim=-1)
        slopes = F.softplus(params[:, :, 2]) + 1e-3
        # Cumulative tone curve: integrate widths/heights
        cdf_x = torch.cumsum(widths, dim=-1)
        cdf_y = torch.cumsum(heights * slopes, dim=-1)
        cdf_x = cdf_x / cdf_x[:, -1:].clamp(min=1e-6)
        cdf_y = cdf_y / cdf_y[:, -1:].clamp(min=1e-6)
        y_flat = y.reshape(b, -1)
        out = []
        for i in range(b):
            xi = cdf_x[i]
            yi = cdf_y[i]
            xf = y_flat[i].clamp(0.0, 1.0)
            idx = torch.searchsorted(xi, xf).clamp(1, self.num_knots - 1)
            x0, x1 = xi[idx - 1], xi[idx]
            y0, y1 = yi[idx - 1], yi[idx]
            t = (xf - x0) / (x1 - x0 + 1e-8)
            out.append(y0 + t * (y1 - y0))
        return torch.stack(out).view_as(y)

    def forward(self, y: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.param_head(y)
        y_hat = self._monotone_map(y, logits)
        uv = self.chroma_refine(torch.cat([u, v], dim=1))
        u_hat, v_hat = uv[:, 0:1], uv[:, 1:2]
        return y_hat, u_hat, v_hat

    def spline_smoothness_loss(self, logits: Tensor) -> Tensor:
        b = logits.shape[0]
        params = logits.view(b, self.num_knots, 3)
        slopes = params[:, :, 2]
        return (slopes[:, 1:] - slopes[:, :-1]).abs().mean()
