"""FDIM hybrid model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.fdim.deep_branch import DeepBranch
from ltx_trainer.fdim.hdr_preprocess import hdr_rgb_to_pu21_input
from ltx_trainer.fdim.mapping import LogisticMapping, fuse_component_scores
from ltx_trainer.fdim.trad_branch import trad_vmaf_proxy


@dataclass
class FDIMConfig:
    use_pu21: bool = False
    l_peak: float = 1000.0
    deep_map: LogisticMapping = field(default_factory=lambda: LogisticMapping(beta3=3.0, beta2=1.0))
    trad_map: LogisticMapping = field(default_factory=lambda: LogisticMapping(beta3=50.0, beta2=0.05))


class FDIM(nn.Module):
    """Hybrid full-reference VQA (arXiv:2604.24123)."""

    def __init__(self, cfg: FDIMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FDIMConfig()
        self.deep = DeepBranch()

    def _prep(self, x: Tensor) -> Tensor:
        if self.cfg.use_pu21:
            return hdr_rgb_to_pu21_input(x, l_peak=self.cfg.l_peak)
        return x

    def score_frame(self, ref: Tensor, dist: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ref_p = self._prep(ref)
        dist_p = self._prep(dist)
        q_deep, _ = self.deep.forward_frame(ref_p, dist_p)
        q_trad = trad_vmaf_proxy(ref_p, dist_p)
        if q_deep.ndim == 0:
            q_deep = q_deep.reshape(1)
        d_map = self.cfg.deep_map(q_deep)
        t_map = self.cfg.trad_map(q_trad)
        return fuse_component_scores(d_map, t_map), d_map, t_map

    def forward(self, ref: Tensor, dist: Tensor) -> Tensor:
        """Video/frame tensor ``[T,3,H,W]`` or single frame ``[3,H,W]``."""
        if ref.dim() == 3:
            score, _, _ = self.score_frame(ref, dist)
            return score
        scores = []
        for t in range(ref.shape[0]):
            s, _, _ = self.score_frame(ref[t], dist[t])
            scores.append(s.reshape(1))
        return torch.cat(scores).mean().reshape(1)

    def deep_only(self, ref: Tensor, dist: Tensor) -> Tensor:
        ref_p = self._prep(ref)
        dist_p = self._prep(dist)
        if ref_p.dim() == 3:
            q, _ = self.deep.forward_frame(ref_p, dist_p)
            return self.cfg.deep_map(q.reshape(1) if q.ndim == 0 else q)
        qs = []
        for t in range(ref_p.shape[0]):
            q, _ = self.deep.forward_frame(ref_p[t], dist[t])
            qs.append(q.reshape(1) if q.ndim == 0 else q)
        return self.cfg.deep_map(torch.cat(qs).mean().reshape(1))
