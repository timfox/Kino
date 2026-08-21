"""Long-short term memory update (Sec. 4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ltx_trainer.streamov.config import StreamOVConfig


@dataclass
class MemoryBank:
    """Bounded streaming memory M_t = M_S ∪ M_L."""

    indices: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    scores: Tensor = field(default_factory=lambda: torch.empty(0))
    modality: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))

    def __len__(self) -> int:
        return int(self.indices.numel())


def _topk_indices(scores: Tensor, k: int) -> Tensor:
    k = min(k, scores.numel())
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.topk(scores, k=k, largest=True).indices


def route_modality(ev: Tensor, ea: Tensor, eav: Tensor, t: int) -> int:
    """0=visual, 1=audio, 2=audio-visual aligned."""
    if float(eav[t]) >= float(ev[t]) and float(eav[t]) >= float(ea[t]):
        return 2
    if float(ev[t]) >= float(ea[t]):
        return 0
    return 1


def update_long_short_memory(
    bank: MemoryBank,
    base_scores: Tensor,
    ev: Tensor,
    ea: Tensor,
    eav: Tensor,
    window_offset: int,
    cfg: StreamOVConfig | None = None,
) -> MemoryBank:
    """Select Top-K_S from current window and refresh Top-K_L from candidates."""
    cfg = cfg or StreamOVConfig()
    device = base_scores.device
    t_len = base_scores.numel()
    local_idx = torch.arange(t_len, device=device) + window_offset
    mod = torch.tensor(
        [route_modality(ev, ea, eav, i) for i in range(t_len)],
        device=device,
        dtype=torch.long,
    )

    short_idx = _topk_indices(base_scores, cfg.short_memory_k)
    short = MemoryBank(
        indices=local_idx[short_idx],
        scores=base_scores[short_idx],
        modality=mod[short_idx],
    )

    outgoing = short
    candidates_idx = torch.cat([bank.indices, outgoing.indices]) if len(bank) else outgoing.indices
    candidates_scores = (
        torch.cat([bank.scores, outgoing.scores]) if len(bank) else outgoing.scores
    )
    candidates_mod = torch.cat([bank.modality, outgoing.modality]) if len(bank) else outgoing.modality

    if candidates_idx.numel() == 0:
        long = MemoryBank()
    else:
        long_sel = _topk_indices(candidates_scores, cfg.long_memory_k)
        long = MemoryBank(
            indices=candidates_idx[long_sel],
            scores=candidates_scores[long_sel],
            modality=candidates_mod[long_sel],
        )

    all_idx = torch.cat([short.indices, long.indices])
    all_scores = torch.cat([short.scores, long.scores])
    all_mod = torch.cat([short.modality, long.modality])
    if all_idx.numel() == 0:
        return MemoryBank()

    uniq, inv = torch.unique(all_idx, sorted=True, return_inverse=True)
    agg_scores = torch.zeros(uniq.numel(), device=device)
    agg_mod = torch.zeros(uniq.numel(), device=device, dtype=torch.long)
    for u in range(uniq.numel()):
        mask = inv == u
        agg_scores[u] = all_scores[mask].max()
        agg_mod[u] = all_mod[mask][all_scores[mask].argmax()]
    order = torch.argsort(uniq)
    return MemoryBank(indices=uniq[order], scores=agg_scores[order], modality=agg_mod[order])
