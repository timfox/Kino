"""Symmetric cross-modal Hard Gumbel gating on the text encoder (Sec. 3.4, Eq. 9–10)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.cmap.config import CMAPConfig


def batch_image_mean(visual: Tensor) -> Tensor:
    """Batch mean image feature ``v̄`` (Sec. 3.4); ``visual`` (B, d) → (d,)."""
    return visual.mean(dim=0)


def hard_gumbel_gate_values(logits: Tensor, tau: float) -> Tensor:
    """Two-way Hard Gumbel–Softmax; returns the **on** probability (second class) with STE.

    ``logits`` (..., 2) → (...,) gate in ``{0, 1}`` forward, gradients via straight-through.
    """
    y = F.gumbel_softmax(logits / tau, hard=True, dim=-1)
    return y[..., 1]


def text_layer_gates_from_vbar(W_task: Tensor, v_bar: Tensor, tau: float) -> Tensor:
    """Eq. (9) for one task: ``W_task`` (L, 2, d), ``v_bar`` (d,) → per-layer gates ``g_txt_l`` (L,)."""
    logits = torch.einsum("lod,d->lo", W_task, v_bar)
    return hard_gumbel_gate_values(logits, tau)


def symmetric_text_gate_param_count(num_tasks: int, n_text_layers: int, embed_dim: int) -> int:
    """Trainable scalar count ``T × L_txt × 2 × d`` (Sec. 3.4)."""
    return num_tasks * n_text_layers * 2 * embed_dim


class SymmetricTextGates(nn.Module):
    """Per-task per-layer ``W_l ∈ R^{2×d}``; forward selects task and returns Hard Gumbel gates (L,)."""

    def __init__(
        self,
        num_tasks: int,
        cfg: CMAPConfig,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.cfg = cfg
        d = cfg.embed_dim
        L = cfg.n_text_layers
        self.W = nn.Parameter(torch.empty(num_tasks, L, 2, d, dtype=dtype or torch.float32))
        nn.init.normal_(self.W, std=0.02)

    def forward(self, task_id: int, v_bar: Tensor) -> Tensor:
        """``v_bar`` (d,) → (L,) gate values."""
        if v_bar.dim() != 1:
            raise ValueError("v_bar must be (d,)")
        return text_layer_gates_from_vbar(self.W[task_id], v_bar, self.cfg.gumbel_temperature)
