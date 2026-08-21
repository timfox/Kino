"""Vector-bank interpolation for latent modulation (Sec. IV-B, Eq. 13)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.nvc_erp_qpa.config import NvcErpQpaConfig


def interpolate_vector(
    bank: Tensor,
    q: float | Tensor,
) -> Tensor:
    """
    Eq. (13): linear blend between Ve[floor q] and Ve[ceil q].

    bank: (q_num, C) learned vectors per discrete quality level.
  q: scalar or per-row quality in [0, q_num-1].
    """
    if bank.dim() != 2:
        raise ValueError("bank must be (q_num, channels)")
    q_num = bank.shape[0]
    if isinstance(q, Tensor):
        q = q.clamp(0.0, float(q_num - 1))
        lo = q.floor().long().clamp(0, q_num - 1)
        hi = q.ceil().long().clamp(0, q_num - 1)
        w_hi = (q - lo.float()).unsqueeze(-1)
        w_lo = 1.0 - w_hi
        v_lo = bank[lo]
        v_hi = bank[hi]
        if v_lo.dim() == 1:
            return w_lo * bank[lo] + w_hi * bank[hi]
        return w_lo * v_lo + w_hi * v_hi
    qf = float(max(0.0, min(q, q_num - 1)))
    lo = int(qf // 1)
    hi = min(lo + 1, q_num - 1)
    if lo == hi:
        return bank[lo]
    w_hi = qf - lo
    return (1.0 - w_hi) * bank[lo] + w_hi * bank[hi]


def quantize_q(q: Tensor) -> Tensor:
    """Ablation: floor-only vector selection (Table I w/o Interpolation)."""
    return q.floor()


class VectorBankModulator(nn.Module):
    """Channel-wise latent modulation v(q̃) ⊙ y per DCVC-RT (Fig. 2)."""

    def __init__(self, cfg: NvcErpQpaConfig, *, bank_role: str = "encoder") -> None:
        super().__init__()
        self.cfg = cfg
        self.bank_role = bank_role
        c = cfg.latent_channels
        self.bank = nn.Parameter(torch.randn(cfg.q_num, c) * 0.02)

    def forward(self, latent: Tensor, q_map: Tensor) -> Tensor:
        """
        latent: B×C×H×W
        q_map: (H,) per-latitude q̃φ
        """
        b, ch, h, w = latent.shape
        if q_map.shape[0] != h:
            q_map = torch.nn.functional.interpolate(
                q_map.view(1, 1, -1, 1),
                size=h,
                mode="linear",
                align_corners=True,
            ).view(h)

        out_rows = []
        for row in range(h):
            q_row = float(q_map[row].item())
            if not self.cfg.use_interpolation:
                q_row = float(int(q_row))
            v = interpolate_vector(self.bank, q_row)
            out_rows.append(latent[:, :, row : row + 1, :] * v.view(1, ch, 1, 1))
        return torch.cat(out_rows, dim=2)
