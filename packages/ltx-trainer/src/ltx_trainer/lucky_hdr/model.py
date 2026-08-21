"""LuckyHDR iterative shift-and-merge model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.lucky_hdr.merge_net import MergeStage
from ltx_trainer.lucky_hdr.shift_net import ShiftStage
from ltx_trainer.lucky_hdr.tonemap import normalize_exposure


@dataclass
class LuckyHdrConfig:
    channels: int = 16
    paper_arxiv: str = "arXiv:2604.19976"


class LuckyHdr(nn.Module):
    """Handheld bracket fusion: short → long with shared shift/merge nets."""

    def __init__(self, cfg: LuckyHdrConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or LuckyHdrConfig()
        self.cfg = cfg
        self.shift = ShiftStage(channels=cfg.channels)
        self.merge = MergeStage(channels=cfg.channels)

    def _pair(
        self,
        acc: Tensor,
        nxt: Tensor,
        *,
        valid_hint: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        warped, shift, valid = self.shift(acc, nxt)
        merged, wb, wa = self.merge(acc, warped, valid=valid if valid_hint is None else valid_hint * valid)
        return merged, shift, valid, wb + wa

    def forward(self, stack: Tensor, evs: list[float] | Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """``stack`` ``[N,3,H,W]`` linear exposures → tone-mapped HDR proxy ``[3,H,W]``."""
        if isinstance(evs, Tensor):
            ev_list = evs.detach().cpu().tolist()
        else:
            ev_list = list(evs)
        ref = float(ev_list[0])
        normed = torch.stack([normalize_exposure(stack[i], ev_list[i], ref) for i in range(stack.shape[0])], dim=0)
        acc = normed[0]
        shifts: list[Tensor] = []
        valids: list[Tensor] = []
        for i in range(1, normed.shape[0]):
            acc, shift, valid, _ = self._pair(acc, normed[i])
            shifts.append(shift)
            valids.append(valid)
        return acc, shifts, valids

    def forward_train(
        self,
        stack: Tensor,
        no_shift_stack: Tensor,
        evs: list[float] | Tensor,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Training path with warp targets from shake-free burst."""
        pred, shifts, _ = self.forward(stack, evs)
        warp_terms: list[Tensor] = []
        ref = float(evs[0] if not isinstance(evs, Tensor) else evs[0].item())
        target = normalize_exposure(no_shift_stack[0], ref, ref)
        for i in range(1, no_shift_stack.shape[0]):
            nxt = normalize_exposure(no_shift_stack[i], float(evs[i]), ref)
            warped, _, _ = self.shift(target, nxt)
            warp_terms.append((warped - nxt).abs().mean())
            target, _, _, _ = self._pair(target, nxt)
        return pred, warp_terms, shifts
