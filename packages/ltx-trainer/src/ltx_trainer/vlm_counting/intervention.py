"""Causal intervention via token masking — § D.1.1, Fig. 11."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.vlm_counting.probing import StonePresenceProbe


def mask_top_k_stone_embeddings(emb: Tensor, probe: StonePresenceProbe, k: int) -> Tensor:
    """Zero out k probe-positive patch embeddings (attention-mask analogue)."""
    out = emb.clone()
    preds = probe(emb.unsqueeze(0) if emb.dim() == 2 else emb)[0]
    idx = torch.nonzero(preds > 0.5, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return out
    k = min(k, int(idx.numel()))
    chosen = idx[torch.randperm(idx.numel())[:k]]
    out[chosen] = 0.0
    return out


def steering_accuracy_curve(
    probe: StonePresenceProbe,
    emb: Tensor,
    ng: int,
    *,
    max_k: int = 5,
) -> list[dict[str, float | int | bool]]:
    """Simulate N'_P = N_G - k after masking k probed stones."""
    rows: list[dict[str, float | int | bool]] = []
    for k in range(1, max_k + 1):
        masked = mask_top_k_stone_embeddings(emb, probe, k)
        nh_after = float(probe.hidden_number(masked.unsqueeze(0)).item())
        expected = ng - k
        # Paper: decoder subtracts masked stones when causal link holds
        predicted = int(round(nh_after))
        acc = predicted == expected
        rows.append({"k": k, "nh_after": round(nh_after, 2), "expected_np": expected, "steering_ok": acc})
    return rows
