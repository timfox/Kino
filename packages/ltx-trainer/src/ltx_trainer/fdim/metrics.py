"""Correlation helpers for VQA evaluation."""

from __future__ import annotations

import math

import torch


def plcc(pred: torch.Tensor, mos: torch.Tensor) -> float:
    p = pred.reshape(-1).float()
    m = mos.reshape(-1).float()
    if p.numel() < 2:
        return 0.0
    p = p - p.mean()
    m = m - m.mean()
    den = p.std() * m.std()
    if den <= 1e-8:
        return 0.0
    return float((p * m).mean() / den)


def srocc(pred: torch.Tensor, mos: torch.Tensor) -> float:
    p = pred.reshape(-1)
    m = mos.reshape(-1)
    if p.numel() < 2:
        return 0.0
    rp = torch.argsort(torch.argsort(p)).float()
    rm = torch.argsort(torch.argsort(m)).float()
    return plcc(rp, rm)
