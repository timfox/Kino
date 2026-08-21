"""CamPVG U-Net surrogate with pose + epipolar branches."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.campvg.config import CamPVGConfig
from ltx_trainer.campvg.epipolar import spherical_epipolar_mask_stub
from ltx_trainer.campvg.epipolar_attention import SphericalEpipolarAttention
from ltx_trainer.campvg.plucker_pano import PanoramicPoseEncoder, trajectory_plucker


class CamPVGStub(nn.Module):
    def __init__(self, cfg: CamPVGConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CamPVGConfig()
        c = self.cfg.hidden_dim
        self.frame_enc = nn.Sequential(
            nn.Conv2d(3, c, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pose_enc = PanoramicPoseEncoder(out_dim=c)
        self.epi_attn = SphericalEpipolarAttention(c)
        self.fuse = nn.Conv2d(c * 2, c, 1)
        self.out = nn.Conv2d(c, 3, 3, padding=1)

    def encode_frame(self, frame: Tensor) -> Tensor:
        return self.frame_enc(frame)

    def forward(
        self,
        frames: Tensor,
        rotations: Tensor,
        translations: Tensor,
    ) -> dict[str, Tensor]:
        """
        frames [B,N,3,H,W]; rotations [B,N,3,3]; translations [B,N,3].
        """
        b, n, _, h, w = frames.shape
        plucker = torch.stack(
            [trajectory_plucker(rotations[bi], translations[bi], h, w) for bi in range(b)],
            dim=0,
        )

        feats = []
        for i in range(n):
            feats.append(self.encode_frame(frames[:, i]))
        feat = torch.stack(feats, dim=1)

        if self.cfg.use_pano_plucker:
            pose_feat = self.pose_enc(plucker)
            if pose_feat.shape[-2:] != feat.shape[-2:]:
                b2, n2, c2, _, _ = pose_feat.shape
                pose_feat = F.interpolate(
                    pose_feat.reshape(b2 * n2, c2, h, w),
                    size=feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).view(b2, n2, c2, *feat.shape[-2:])
            feat = feat + pose_feat

        if self.cfg.use_spherical_epipolar and n > 1:
            enhanced = []
            for bi in range(b):
                per = []
                for qi in range(n):
                    q = feat[bi, qi].flatten(1).transpose(0, 1)
                    kv_list = [feat[bi, j].flatten(1).transpose(0, 1) for j in range(n)]
                    kv = torch.stack(kv_list, dim=0)
                    hw = q.shape[0]
                    mask = torch.zeros(hw, n * hw, device=frames.device)
                    for j in range(n):
                        if j == qi:
                            mask[:, j * hw : (j + 1) * hw] = torch.eye(hw, device=frames.device)
                        else:
                            m2d = spherical_epipolar_mask_stub(
                                feat.shape[-2],
                                feat.shape[-1],
                                rotations[bi, qi],
                                translations[bi, qi],
                                rotations[bi, j],
                                translations[bi, j],
                                k=min(self.cfg.epipolar_k, 32),
                            )
                            mask[:, j * hw : (j + 1) * hw] = m2d.reshape(-1).unsqueeze(0).expand(hw, -1)
                    q_out = self.epi_attn(q, kv, mask)
                    per.append(q_out.transpose(0, 1).reshape(-1, feat.shape[-2], feat.shape[-1]))
                enhanced.append(torch.stack(per, dim=0))
            epi_feat = torch.stack(enhanced, dim=0)
            fused = []
            for i in range(n):
                fused.append(self.fuse(torch.cat([feat[:, i], epi_feat[:, i]], dim=1)))
            feat = torch.stack(fused, dim=1)

        recon = torch.stack([self.out(feat[:, i]) for i in range(n)], dim=1)
        if recon.shape[-2:] != (h, w):
            b2, n2, c2 = recon.shape[:3]
            recon = F.interpolate(
                recon.reshape(b2 * n2, c2, *recon.shape[-2:]),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).view(b2, n2, c2, h, w)
        return {"frames": recon, "plucker": plucker}
