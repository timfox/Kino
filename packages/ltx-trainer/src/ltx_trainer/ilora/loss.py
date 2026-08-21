"""Training objective — Eq. (14)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def prediction_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return nn.functional.cross_entropy(logits, labels)


def ilora_total_loss(
    pred_loss: Tensor,
    pois_kl: Tensor,
    lap_kl: Tensor,
    *,
    lambda_pois: float,
    lambda_lap: float,
) -> Tensor:
    return pred_loss + lambda_pois * pois_kl + lambda_lap * lap_kl
