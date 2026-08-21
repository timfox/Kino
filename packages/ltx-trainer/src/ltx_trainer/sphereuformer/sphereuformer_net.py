"""SphereUFormer U-shaped SAM network stub (Fig. 4)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.sphereuformer.config import NUM_ENCODER_STAGES, SphereUFormerConfig
from ltx_trainer.sphereuformer.icosphere import center_pool, nearest_up, node_count
from ltx_trainer.sphereuformer.positional import VerticalGlobalPE
from ltx_trainer.sphereuformer.slsa import SphericalAttentionBlock


def _phi_coords(n: int, device: torch.device) -> Tensor:
    return torch.linspace(0, math.pi, n, device=device)


class SphereUFormerStub(nn.Module):
    def __init__(self, cfg: SphereUFormerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SphereUFormerConfig()
        n0 = node_count(self.cfg.rank, node_type=self.cfg.node_type)
        self.n0 = min(n0, 512)  # cap for CPU stub
        d0 = self.cfg.embed_dim
        self.input_proj = nn.Linear(self.cfg.in_ch, d0)
        self.init_pe = VerticalGlobalPE(d0)
        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.down_projs = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        self.dec_fuse = nn.ModuleList()
        ns = [self.n0]
        dims = [d0]
        for s in range(NUM_ENCODER_STAGES):
            n = max(12, ns[-1] // 4)
            ns.append(n)
            dims.append(d0 * (2 ** (s + 1)))
        self.stage_nodes = ns
        self.stage_dims = dims
        for s in range(NUM_ENCODER_STAGES):
            n, d = ns[s], dims[s]
            heads = 2 ** (s + 1)
            self.enc_blocks.append(
                nn.ModuleList(
                    [
                        SphericalAttentionBlock(
                            d,
                            num_heads=heads,
                            c_head=self.cfg.c_head,
                            c_win=self.cfg.c_win,
                            n_nodes=n,
                        )
                        for _ in range(2)
                    ]
                )
            )
            self.down_projs.append(nn.Linear(d, dims[s + 1]))
        bn, bd = ns[-1], dims[-1]
        self.bottleneck = nn.ModuleList(
            [
                SphericalAttentionBlock(
                    bd, num_heads=16, c_head=self.cfg.c_head, c_win=self.cfg.c_win, n_nodes=bn
                )
                for _ in range(2)
            ]
        )
        for s in reversed(range(NUM_ENCODER_STAGES)):
            n, d = ns[s], dims[s]
            d_in = dims[s + 1]
            heads = 2 ** (s + 2)
            self.up_projs.append(nn.Linear(d_in, d))
            self.dec_fuse.append(nn.Linear(d * 2, d))
            self.dec_blocks.append(
                nn.ModuleList(
                    [
                        SphericalAttentionBlock(
                            d,
                            num_heads=min(heads, 16),
                            c_head=self.cfg.c_head,
                            c_win=self.cfg.c_win,
                            n_nodes=n,
                        )
                        for _ in range(2)
                    ]
                )
            )
        out_ch = self.cfg.out_ch_depth if self.cfg.task == "depth" else self.cfg.num_classes_seg
        self.output_proj = nn.Linear(d0, out_ch)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # x: B x C x H x W ERP stub → flatten to nodes
        b, c, h, w = x.shape
        n = min(self.n0, h * w)
        raw = x.view(b, c, -1).transpose(1, 2)[:, :n, : self.cfg.in_ch]
        feats = self.input_proj(raw)
        phi = _phi_coords(n, feats.device)
        feats = feats + self.init_pe(phi).unsqueeze(0)
        skips: list[Tensor] = []
        h_cur = feats
        for stage, blocks in enumerate(self.enc_blocks):
            for blk in blocks:
                h_cur = blk(h_cur, phi[: h_cur.shape[1]])
            skips.append(h_cur)
            h_cur = center_pool(h_cur)
            h_cur = self.down_projs[stage](h_cur)
            phi = _phi_coords(h_cur.shape[1], h_cur.device)
        for blk in self.bottleneck:
            h_cur = blk(h_cur, phi)
        for i, blocks in enumerate(self.dec_blocks):
            s = NUM_ENCODER_STAGES - 1 - i
            target_n = skips[s].shape[1]
            h_cur = nearest_up(h_cur, target_n)
            h_cur = self.up_projs[i](h_cur)
            h_cur = self.dec_fuse[i](torch.cat([h_cur, skips[s]], dim=-1))
            phi = _phi_coords(h_cur.shape[1], h_cur.device)
            for blk in blocks:
                h_cur = blk(h_cur, phi)
        out = self.output_proj(h_cur)
        if self.cfg.task == "depth":
            out = torch.relu(out.squeeze(-1))
            nn = out.shape[1]
            depth_erp = torch.nn.functional.interpolate(
                out.view(b, 1, 1, nn), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)
            return {"depth": depth_erp, "nodes": out}
        logits = out
        return {"logits": logits, "seg": logits.argmax(dim=-1)}
