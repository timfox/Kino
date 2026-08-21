"""Deterministic phase transition at L_crit (Sec. 4.3)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


L_CRIT: float = math.log(2.0)


def is_ordered_phase(token_loss: Tensor, l_crit: float = L_CRIT) -> Tensor:
    """L < L_crit ⇒ P_target > 0.5 under greedy decoding."""
    return token_loss < l_crit


def memft_threshold_weight(token_loss: Tensor, l_crit: float = L_CRIT) -> Tensor:
    """MemFT-OT hard mask (Eq. 9)."""
    return (token_loss > l_crit).to(token_loss.dtype)


def memft_soft_weight(token_loss: Tensor, l_crit: float = L_CRIT, kappa: float = 10.0) -> Tensor:
    """Soft threshold via sigmoid for MemFT-SW base weight."""
    return torch.sigmoid(kappa * (token_loss - l_crit))


def greedy_correct(pred: Tensor, target: Tensor) -> Tensor:
    return (pred == target).to(torch.float32)


def token_accuracy(pred: Tensor, target: Tensor) -> float:
    return float(greedy_correct(pred, target).mean().item())


def exact_match_accuracy(pred: Tensor, target: Tensor) -> float:
    return float((pred == target).all().item())
