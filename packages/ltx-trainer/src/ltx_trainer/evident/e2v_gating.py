"""Entity-to-eVidence gating (Sec. 4.4, Eq. 6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def minmax_normalize(scores: Tensor, dim: int = 0) -> Tensor:
    lo = scores.amin(dim=dim, keepdim=True)
    hi = scores.amax(dim=dim, keepdim=True)
    return (scores - lo) / (hi - lo).clamp(min=1e-6)


def frame_entity_similarity(
    frame_feats: Tensor,
    query_sub: Tensor,
    query_obj: Tensor | None,
) -> Tensor:
    """Cosine similarity per frame; ``frame_feats`` (T, d), queries (d,)."""
    frame_n = F.normalize(frame_feats, dim=-1)
    sub_n = F.normalize(query_sub.unsqueeze(0), dim=-1)
    s_sub = (frame_n * sub_n).sum(dim=-1)
    if query_obj is None:
        return minmax_normalize(s_sub, dim=0)
    obj_n = F.normalize(query_obj.unsqueeze(0), dim=-1)
    s_obj = (frame_n * obj_n).sum(dim=-1)
    return minmax_normalize(s_sub, dim=0), minmax_normalize(s_obj, dim=0)


def e2v_gating_scores(
    frame_feats: Tensor,
    query_sub: Tensor,
    query_obj: Tensor | None = None,
) -> Tensor:
    """Eq. (6): g_t = ŝ_sub · ŝ_obj; absent concept passes through as 1."""
    if query_obj is None:
        sim = frame_entity_similarity(frame_feats, query_sub, None)
        return sim
    s_sub, s_obj = frame_entity_similarity(frame_feats, query_sub, query_obj)
    return s_sub * s_obj
