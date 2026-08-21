"""Forward / backward video OCL streams — Eq. (1), (3), (7)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ltx_trainer.ocl_icc.config import OCLICCConfig
from ltx_trainer.ocl_icc.slots import SlotAttention, TransitionModule


@dataclass
class StreamOutput:
    slots: list[Tensor]
    reconstructions: list[Tensor]
    attentions: list[Tensor]


class VideoOCLStreams(nn.Module):
    """Shared aggregator/decoder; separate forward/backward transitioners."""

    def __init__(self, cfg: OCLICCConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or OCLICCConfig()
        self.cfg = cfg
        self.aggregator = SlotAttention(cfg.num_slots, cfg.slot_dim, cfg.feature_dim, cfg.slot_attn_iters)
        self.transition_fw = TransitionModule(cfg.slot_dim, cfg.feature_dim, backward=False)
        self.transition_bw = TransitionModule(cfg.slot_dim, cfg.feature_dim, backward=True)
        self.init_queries = nn.Parameter(torch.randn(1, cfg.num_slots, cfg.slot_dim) * 0.02)

    def forward_stream(self, frames: Tensor) -> StreamOutput:
        """frames [T, B, N, D]"""
        t_len = frames.shape[0]
        slots_list: list[Tensor] = []
        recon_list: list[Tensor] = []
        attn_list: list[Tensor] = []
        prev = None
        for t in range(t_len):
            feat = frames[t]
            if t == 0:
                q = self.init_queries.expand(feat.shape[0], -1, -1)
            else:
                assert prev is not None
                q = self.transition_fw(prev, feat, noise_std=self.cfg.transition_noise_std)
            s, a = self.aggregator(feat, q)
            slots_list.append(s)
            recon_list.append(self.aggregator.decode(s))
            attn_list.append(a)
            prev = s
        return StreamOutput(slots_list, recon_list, attn_list)

    def backward_stream(self, frames: Tensor, last_forward_slots: Tensor) -> StreamOutput:
        """Process t = T-1 .. 0; init from forward S_T — Eq. (7)."""
        t_len = frames.shape[0]
        slots_list: list[Tensor] = []
        recon_list: list[Tensor] = []
        attn_list: list[Tensor] = []
        prev = None
        for rev_t in range(t_len):
            t = t_len - 1 - rev_t
            feat = frames[t]
            if rev_t == 0:
                q = self.transition_bw(last_forward_slots, feat, noise_std=self.cfg.transition_noise_std)
            else:
                assert prev is not None
                q = self.transition_bw(prev, feat, noise_std=self.cfg.transition_noise_std)
            s, a = self.aggregator(feat, q)
            slots_list.insert(0, s)
            recon_list.insert(0, self.aggregator.decode(s))
            attn_list.insert(0, a)
            prev = s
        return StreamOutput(slots_list, recon_list, attn_list)

    def run_bidirectional(self, frames: Tensor) -> tuple[StreamOutput, StreamOutput]:
        fw = self.forward_stream(frames)
        bw = self.backward_stream(frames, fw.slots[-1])
        return fw, bw
