"""High-level wiring for F-RNG trainable blocks (paper Fig. 2).

Frozen **RelitLRM** and **DiffusionRenderer** are not invoked: pass their latent tensors (or synthetic
tensors in tests). This module connects saliency → fine geometry synthesis → MaterialFormer →
optional neural appearance decode for deferred shading prototypes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ltx_trainer.frng.config import FRNGConfig
from ltx_trainer.frng.decoder import UniversalNeuralAppearanceDecoder
from ltx_trainer.frng.geometry import FineGeometrySynthesis
from ltx_trainer.frng.material_former import MaterialFormer, light_independence_loss
from ltx_trainer.frng.saliency import patch_grid_shape, saliency_map, saliency_patch_scores, topk_patch_flat_indices


@dataclass
class FRNGViews:
    """One object with ``V`` views of RGB (or linear RGB) images."""

    images: Tensor
    """(B, V, 3, H, W)"""


def rgb_to_gray(images: Tensor) -> Tensor:
    """(B, V, 3, H, W) → (B*V, 1, H, W) luminance."""
    b, v, c, h, w = images.shape
    if c != 3:
        raise ValueError("expected 3 channels")
    x = images.reshape(b * v, c, h, w)
    r, g, b_ = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b_


class FRNGTrainableStack(nn.Module):
    """Bundles fine geometry synthesis, MaterialFormer, and universal appearance decoder."""

    def __init__(
        self,
        cfg: FRNGConfig,
        *,
        idm_patch_channels: int,
        fine_geo_out_dim: int,
        mat_feat_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.fine_geom = FineGeometrySynthesis(cfg, idm_in_channels=idm_patch_channels, fine_out_dim=fine_geo_out_dim)
        self.material_former = MaterialFormer(cfg, mat_out_dim=mat_feat_dim)

    def salient_indices_from_views(self, images: Tensor, ph: int, pw: int) -> Tensor:
        """Pick salient patches using mean saliency across views; returns (B, K) flat indices."""
        gray = rgb_to_gray(images)
        s = saliency_map(
            gray,
            box_kernel=self.cfg.box_kernel,
            gaussian_sigma=self.cfg.gaussian_sigma,
            edge_attenuation=self.cfg.edge_attenuation,
        )
        scores = saliency_patch_scores(s, self.cfg.patch_size)
        # average over views: (B, V, Ph, Pw) if we reshape — here gray is B*V, fold back
        b, v, _, h, w = images.shape
        sc = scores.view(b, v, ph, pw).mean(dim=1)
        return topk_patch_flat_indices(sc, self.cfg.saliency_top_frac)

    def forward_material(
        self,
        t_geo: Tensor,
        t_app: Tensor,
        t_prior: Tensor,
        t_app_alt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Material path. If ``t_app_alt`` is given, returns ``l_ind`` scalar; else ``None``."""
        t_mat, g_mat = self.material_former(t_geo, t_app, t_prior)
        l_ind = None
        if t_app_alt is not None:
            t_mat2, _ = self.material_former(t_geo, t_app_alt, t_prior)
            l_ind = light_independence_loss(
                t_mat,
                t_mat2,
                lambda_cos=self.cfg.lambda_cos,
                lambda_kld=self.cfg.lambda_kld,
            )
        return t_mat, g_mat, l_ind


class FRNGComplete(nn.Module):
    """Optional one-file demo: saliency + fine geometry + material + neural decoder."""

    def __init__(
        self,
        cfg: FRNGConfig,
        *,
        idm_patch_channels: int,
        fine_geo_out_dim: int,
        mat_feat_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stack = FRNGTrainableStack(
            cfg,
            idm_patch_channels=idm_patch_channels,
            fine_geo_out_dim=fine_geo_out_dim,
            mat_feat_dim=mat_feat_dim,
        )
        self.decoder = UniversalNeuralAppearanceDecoder(cfg, mat_feat_dim=mat_feat_dim)

    def forward(
        self,
        images: Tensor,
        t_geo: Tensor,
        t_app: Tensor,
        t_prior: Tensor,
        idm_patch_features: Tensor,
        omega_o: Tensor,
        omega_i: Tensor,
        t_app_alt: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """``images`` (B,V,3,H,W) for saliency; tokens (B,P,D); directions (B,R,3) for R ray samples."""
        b, v, c, h, w = images.shape
        ph, pw = patch_grid_shape(h, w, self.cfg.patch_size)
        idx = self.stack.salient_indices_from_views(images, ph, pw)
        fine = self.stack.fine_geom(t_geo, idm_patch_features, idx, ph=ph, pw=pw)
        t_mat, g_mat, l_ind = self.stack.forward_material(t_geo, t_app, t_prior, t_app_alt=t_app_alt)
        # decode subset of rays: use mean material vector
        g = g_mat.mean(dim=1, keepdim=True).expand(-1, omega_o.shape[1], -1)
        rgb = self.decoder(g, omega_o, omega_i)
        out: dict[str, Tensor] = {"fine_geometry": fine, "g_mat": g_mat, "t_mat": t_mat, "rgb": rgb}
        if l_ind is not None:
            out["l_ind"] = l_ind
        return out
