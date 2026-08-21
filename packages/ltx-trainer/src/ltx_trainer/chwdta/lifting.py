"""Learnable channel-wise lifting wavelet WTc / IWTc (Eq. 17, CDF 9/7 init)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# CDF 9/7-style initialization (paper §III-C)
CDF97_ALPHA = -1.5861343425299948
CDF97_BETA = -0.0529801185722639
CDF97_GAMMA = 0.8829110755648989
CDF97_DELTA = 0.4435068520431039
CDF97_K = 1.2301741055862352


class LearnableLifting1D(nn.Module):
    """Four-step predict/update lifting along channel axis for even C."""

    def __init__(self, channels: int, learnable: bool = True) -> None:
        super().__init__()
        assert channels % 2 == 0
        self.channels = channels
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(CDF97_ALPHA))
            self.beta = nn.Parameter(torch.tensor(CDF97_BETA))
            self.gamma = nn.Parameter(torch.tensor(CDF97_GAMMA))
            self.delta = nn.Parameter(torch.tensor(CDF97_DELTA))
            self.kappa = nn.Parameter(torch.tensor(0.0))  # K = exp(kappa)
        else:
            self.register_buffer("alpha", torch.tensor(CDF97_ALPHA))
            self.register_buffer("beta", torch.tensor(CDF97_BETA))
            self.register_buffer("gamma", torch.tensor(CDF97_GAMMA))
            self.register_buffer("delta", torch.tensor(CDF97_DELTA))
            self.register_buffer("kappa", torch.tensor(0.0))

    def _neighbor_sum_even(self, x: Tensor) -> Tensor:
        # x: B, C_even, H, W — circular neighbor sum along channel pairs
        return x + torch.roll(x, shifts=1, dims=1)

    def _neighbor_sum_odd(self, x: Tensor) -> Tensor:
        return x + torch.roll(x, shifts=-1, dims=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: B×C×H×W → smooth S, detail D branches (half channels each)."""
        b, c, h, w = x.shape
        s = x[:, 0::2]
        d = x[:, 1::2]
        d = d + self.alpha * self._neighbor_sum_even(s)
        s = s + self.beta * self._neighbor_sum_odd(d)
        d = d + self.gamma * self._neighbor_sum_even(s)
        s = s + self.delta * self._neighbor_sum_odd(d)
        k = torch.exp(self.kappa)
        s = s * k
        d = d / k
        return s, d

    def inverse(self, s: Tensor, d: Tensor) -> Tensor:
        k = torch.exp(self.kappa)
        s = s / k
        d = d * k
        s = s - self.delta * self._neighbor_sum_odd(d)
        d = d - self.gamma * self._neighbor_sum_even(s)
        s = s - self.beta * self._neighbor_sum_odd(d)
        d = d - self.alpha * self._neighbor_sum_even(s)
        even = s
        odd = d
        out = torch.zeros(s.shape[0], s.shape[1] * 2, s.shape[2], s.shape[3], device=s.device, dtype=s.dtype)
        out[:, 0::2] = even
        out[:, 1::2] = odd
        return out


def wtc(x: Tensor, lift: LearnableLifting1D) -> Tensor:
    """WTc: concat(S, D) along channel dim."""
    s, d = lift(x)
    return torch.cat([s, d], dim=1)


def iwtc(y: Tensor, lift: LearnableLifting1D) -> Tensor:
    c = y.shape[1]
    return lift.inverse(y[:, : c // 2], y[:, c // 2 :])
