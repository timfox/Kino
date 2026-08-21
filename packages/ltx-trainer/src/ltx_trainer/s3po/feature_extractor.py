"""360° feature extractor with CBAM-style attention (Eq. 1–2, Fig. 5)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PanoramaFeatureExtractor(nn.Module):
    """Co-joint features from Ft−1, Ft, Ft+1 + spatial/channel attention."""

    def __init__(self, dim: int, *, use_attention: bool = True) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.conv = nn.Conv2d(3, dim, 3, padding=1)
        self.joint = nn.Conv2d(dim, dim, 3, padding=1)
        self.local1 = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.local2 = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.fuse = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.spatial_att = nn.Conv2d(2, 1, 7, padding=3)
        self.ca_pool = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
        )

    def _frame_feat(self, x: Tensor) -> Tensor:
        return F.relu(self.conv(x))

    def forward(self, prev_f: Tensor, target: Tensor, next_f: Tensor) -> Tensor:
        f_m, f_t, f_p = self._frame_feat(prev_f), self._frame_feat(target), self._frame_feat(next_f)
        joint = F.relu(self.joint(f_m + f_t + f_p))
        corr_m = joint + f_m
        corr_t = joint + f_t
        corr_p = joint + f_p
        local1 = F.relu(self.local1(torch.cat([corr_m, corr_t], dim=1)))
        local2 = F.relu(self.local2(torch.cat([corr_p, corr_t], dim=1)))
        feat = F.relu(self.fuse(torch.cat([local1, local2], dim=1)))
        if not self.use_attention:
            return feat
        spa_in = torch.cat(
            [feat.amax(dim=1, keepdim=True), feat.mean(dim=1, keepdim=True)],
            dim=1,
        )
        spa = torch.sigmoid(self.spatial_att(spa_in))
        feat = feat * spa
        ca = torch.sigmoid(
            self.ca_pool(feat.mean(dim=(2, 3), keepdim=True))
            + self.ca_pool(feat.amax(dim=3, keepdim=True).amax(dim=2, keepdim=True))
        )
        return feat * ca
