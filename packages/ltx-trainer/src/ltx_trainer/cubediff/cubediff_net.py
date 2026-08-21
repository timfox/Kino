"""CubeDiff VAE + inflated LDM stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.cubediff.config import CubeDiffConfig
from ltx_trainer.cubediff.cubemap import cubemap_to_erp, erp_to_cubemap_faces
from ltx_trainer.cubediff.inflated_attn import InflatedCrossAttention, InflatedSelfAttention
from ltx_trainer.cubediff.positional import cube_uv_encoding
from ltx_trainer.cubediff.synced_gn import SyncedGroupNorm


class CubeDiffStub(nn.Module):
    def __init__(self, cfg: CubeDiffConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CubeDiffConfig()
        t, c = self.cfg.num_faces, 4
        self.enc = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1),
            SyncedGroupNorm(2, c) if self.cfg.use_synced_gn else nn.GroupNorm(2, c),
            nn.SiLU(),
        )
        self.self_attn = InflatedSelfAttention(c) if self.cfg.use_inflated_attn else nn.Identity()
        self.cross_attn = InflatedCrossAttention(c, 64)
        self.text_proj = nn.Linear(64, 64)
        self.dec = nn.Conv2d(c + (2 if self.cfg.use_pos_encoding else 0), 3, 3, padding=1)

    def _add_pos(self, faces: Tensor) -> Tensor:
        if not self.cfg.use_pos_encoding:
            return faces
        b, t, c, h, w = faces.shape
        pe = torch.cat([cube_uv_encoding(h, w, i, device=faces.device) for i in range(t)], dim=0)
        pe = pe.unsqueeze(0).expand(b, t, -1, h, w)
        return torch.cat([faces, pe], dim=2)

    def forward(
        self,
        erp: Tensor | None = None,
        *,
        cond_face: Tensor | None = None,
        text_emb: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if erp is None:
            erp = torch.randn(1, 3, 256, 512)
        fh = min(64, erp.shape[-2] // 4)
        fw = min(64, erp.shape[-1] // 8)
        faces = erp_to_cubemap_faces(erp, fh, fw)
        if cond_face is not None:
            faces[:, 0] = torch.nn.functional.interpolate(
                cond_face, size=(fh, fw), mode="bilinear", align_corners=False
            )
        b, t, c_in, fh, fw = faces.shape
        z = self.enc(faces.reshape(b * t, c_in, fh, fw)).reshape(b, t, -1, fh, fw)
        if isinstance(self.self_attn, InflatedSelfAttention):
            z = self.self_attn(z)
        ctx = self.text_proj(text_emb) if text_emb is not None else torch.zeros(z.shape[0], 64, device=z.device)
        if isinstance(self.cross_attn, InflatedCrossAttention):
            z = self.cross_attn(z, ctx)
        z = self._add_pos(z)
        b, t, cz, fh, fw = z.shape
        rgb_flat = self.dec(z.reshape(b * t, cz, fh, fw))
        rgb_faces = rgb_flat.reshape(b, t, 3, fh, fw)
        pano = cubemap_to_erp(rgb_faces, erp.shape[-2], erp.shape[-1])
        return {"faces": rgb_faces, "panorama": pano}
