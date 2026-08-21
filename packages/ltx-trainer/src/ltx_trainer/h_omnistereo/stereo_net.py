"""Side-tuning + iterative omnidirectional stereo matching (Sec. III-C)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.h_omnistereo.config import HOmniStereoConfig
from ltx_trainer.h_omnistereo.normal_net import HeadingAlignedNormalNet


class SideContextCNN(nn.Module):
    """Lightweight context network (image-only branch)."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SideTuningAdapter(nn.Module):
    """Downscale normal priors (4×4, stride 4) and concat with side CNN features."""

    def __init__(self, prior_dim: int, side_dim: int) -> None:
        super().__init__()
        self.prior_down = nn.Conv2d(prior_dim, side_dim, kernel_size=4, stride=4)
        self.merge = nn.Conv2d(side_dim * 2, side_dim, 3, padding=1)

    def forward(self, priors: Tensor, side: Tensor) -> Tensor:
        p = self.prior_down(priors)
        if p.shape[-2:] != side.shape[-2:]:
            p = torch.nn.functional.interpolate(p, size=side.shape[-2:], mode="bilinear", align_corners=False)
        return self.merge(torch.cat([p, side], dim=1))


class AttentiveHybridCostFilter(nn.Module):
    """Aggregate 4D hybrid cost volume (spatial + disparity groups)."""

    def __init__(self, groups: int, dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(groups, groups, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(groups, 1, 1),
        )
        self.out = nn.Conv2d(1, dim, 3, padding=1)

    def forward(self, cost4d: Tensor) -> Tensor:
        # cost4d: [B, G, D, H, W] -> context [B, dim, H, W]
        filtered = self.conv(cost4d).squeeze(1)  # [B, D, H, W]
        if filtered.dim() == 4:
            filtered = filtered.mean(dim=1, keepdim=True)
        return self.out(filtered)


class ConvGRUCell(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv_z = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.conv_r = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.conv_h = nn.Conv2d(dim * 2, dim, 3, padding=1)

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.conv_z(hx))
        r = torch.sigmoid(self.conv_r(hx))
        h_tilde = torch.tanh(self.conv_h(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * h_tilde


class HOmniStereoNet(nn.Module):
    """Top-bottom ERP stereo: shared normal priors + iterative disparity + uncertainty."""

    def __init__(self, cfg: HOmniStereoConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HOmniStereoConfig()
        d = self.cfg.feature_dim
        self.normal_net = HeadingAlignedNormalNet(self.cfg)
        self.side_top = SideContextCNN(d)
        self.side_bottom = SideContextCNN(d)
        self.adapter_top = SideTuningAdapter(d, d)
        self.adapter_bottom = SideTuningAdapter(d, d)
        self.corr_proj = nn.Conv2d(d * 2, d, 1)
        self.cost_filter = AttentiveHybridCostFilter(self.cfg.cost_groups, d)
        self.disp_init = nn.Conv2d(d, 1, 3, padding=1)
        self.disp_in = nn.Conv2d(d + 1, d, 1)
        self.unc_in = nn.Conv2d(d + 1, d, 1)
        self.disp_gru = ConvGRUCell(d)
        self.unc_gru = ConvGRUCell(d)
        self.disp_delta = nn.Conv2d(d, 1, 3, padding=1)
        self.unc_delta = nn.Conv2d(d, 1, 3, padding=1)

    def _hybrid_features(self, image: Tensor, *, view: str) -> Tensor:
        priors = self.normal_net.encode_priors(image)
        side = self.side_top(image) if view == "top" else self.side_bottom(image)
        adapter = self.adapter_top if view == "top" else self.adapter_bottom
        return adapter(priors, side)

    def _build_cost_volume(self, feat_top: Tensor, feat_bot: Tensor, max_disp: int) -> Tensor:
        b, _c, h, w = feat_top.shape
        d = min(max_disp, w)
        vol = []
        for disp in range(d):
            shifted = torch.roll(feat_bot, shifts=-disp, dims=3)
            corr = (feat_top * shifted).sum(dim=1, keepdim=True)
            vol.append(corr)
        cost = torch.stack(vol, dim=2)  # [B, 1, D, H, W]
        g = self.cfg.cost_groups
        return cost.expand(b, g, d, h, w)

    def forward(
        self,
        top: Tensor,
        bottom: Tensor,
        *,
        refine_iters: int | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
        iters = refine_iters if refine_iters is not None else self.cfg.refine_iters
        full_hw = top.shape[-2:]
        ft = self._hybrid_features(top, view="top")
        fb = self._hybrid_features(bottom, view="bottom")
        fused = self.corr_proj(torch.cat([ft, fb], dim=1))
        cost4d = self._build_cost_volume(ft, fb, self.cfg.max_disparity)
        ctx = self.cost_filter(cost4d)
        disp = self.disp_init(ctx)

        def _upsample(d: Tensor) -> Tensor:
            if d.shape[-2:] == full_hw:
                return d
            return torch.nn.functional.interpolate(d, size=full_hw, mode="bilinear", align_corners=False)

        disparities: list[Tensor] = [_upsample(disp)]
        uncertainties: list[Tensor] = []
        h_d = torch.zeros_like(fused)
        h_u = torch.zeros_like(fused)
        for _ in range(iters):
            ctx_in = torch.cat([fused, disp], dim=1)
            h_d = self.disp_gru(h_d, self.disp_in(ctx_in))
            disp = disp + self.disp_delta(h_d)
            disparities.append(_upsample(disp))
            h_u = self.unc_gru(h_u, self.unc_in(ctx_in))
            u = self.unc_delta(h_u)
            uncertainties.append(_upsample(torch.exp(u.clamp(-5, 5))))
        return {
            "disparity": disparities[-1],
            "disparities": disparities,
            "uncertainties": uncertainties,
            "feat_top": ft,
            "feat_bottom": fb,
        }
