"""Flow-matching training loss (Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class FlowMatchingLoss:
    """Rectified flow: ``z_t = (1-t)ε + t z_tgt``, supervise ``v = z_tgt - ε``."""

    def __call__(
        self,
        net,
        z_tgt: Tensor,
        z_ref: Tensor,
        *,
        t: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        if z_tgt.dim() == 3:
            z_tgt = z_tgt.unsqueeze(0)
            z_ref = z_ref.unsqueeze(0)
        b = z_tgt.shape[0]
        eps = torch.randn_like(z_tgt)
        if t is None:
            t = torch.rand(b, device=z_tgt.device, dtype=z_tgt.dtype)
        elif t.ndim == 0:
            t = t.reshape(1).expand(b)
        t_view = t.view(b, 1, 1, 1)
        z_t = (1.0 - t_view) * eps + t_view * z_tgt
        v_tgt = z_tgt - eps
        v_pred = net(z_t, z_ref, t)
        loss = F.mse_loss(v_pred, v_tgt)
        return loss, {"loss_flow": float(loss.detach())}
