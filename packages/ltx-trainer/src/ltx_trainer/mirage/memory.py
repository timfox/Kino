"""Latent spatial memory M = {(p_i, f_i)} and RGB baseline (arXiv:2606.09828)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ltx_trainer.mirage.geometry import (
    backproject_cell,
    downsample_depth,
    latent_intrinsics,
    project_point,
)


@dataclass
class LatentSpatialMemory:
    """Persistent 3D cache storing VAE latent tokens at world locations."""

    channels: int
    points: list[np.ndarray] = field(default_factory=list)
    features: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def clear(self) -> None:
        self.points.clear()
        self.features.clear()

    def add_from_latent(
        self,
        latent: np.ndarray,
        depth: np.ndarray,
        k_pixel: np.ndarray,
        extrinsics: np.ndarray,
        *,
        rgb_hw: tuple[int, int],
        dynamic_mask: np.ndarray | None = None,
        depth_mode: str = "bilinear",
    ) -> int:
        """Lift latent cells into the cache via depth-guided back-projection (Eq. 4)."""
        z = np.asarray(latent, dtype=np.float64)
        if z.ndim != 3:
            raise ValueError(f"expected C×h×w latent, got {z.shape}")
        c, h, w = z.shape
        if c != self.channels:
            raise ValueError(f"latent channels {c} != memory channels {self.channels}")
        d_lat = downsample_depth(depth, (h, w), mode=depth_mode)
        k_l = latent_intrinsics(k_pixel, rgb_hw=rgb_hw, latent_hw=(h, w))
        added = 0
        for v in range(h):
            for u in range(w):
                if dynamic_mask is not None and bool(dynamic_mask[v, u]):
                    continue
                d = float(d_lat[v, u])
                if not np.isfinite(d) or d <= 0:
                    continue
                p = backproject_cell(u, v, d, k_l, extrinsics)
                if not np.all(np.isfinite(p)):
                    continue
                self.points.append(p)
                self.features.append(z[:, v, u].copy())
                added += 1
        return added

    def update_union(
        self,
        latent: np.ndarray,
        depth: np.ndarray,
        k_pixel: np.ndarray,
        extrinsics: np.ndarray,
        *,
        rgb_hw: tuple[int, int],
        dynamic_mask: np.ndarray | None = None,
        depth_mode: str = "bilinear",
    ) -> int:
        """Autoregressive cache update M ← M ∪ {(p_uv, F_uv)} (Eq. 6)."""
        return self.add_from_latent(
            latent,
            depth,
            k_pixel,
            extrinsics,
            rgb_hw=rgb_hw,
            dynamic_mask=dynamic_mask,
            depth_mode=depth_mode,
        )

    def readout(
        self,
        k_pixel: np.ndarray,
        extrinsics: np.ndarray,
        *,
        rgb_hw: tuple[int, int],
        latent_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Latent-resolution z-buffer readout (Eq. 5)."""
        h, w = latent_hw
        z_hat = np.zeros((self.channels, h, w), dtype=np.float64)
        mask = np.zeros((h, w), dtype=np.float64)
        if not self.points:
            return z_hat, mask
        k_l = latent_intrinsics(k_pixel, rgb_hw=rgb_hw, latent_hw=latent_hw)
        # Single pass over cache points (Eq. 10 via z-buffer buckets; equivalent to per-cell omega_set).
        buckets: dict[tuple[int, int], list[tuple[float, int]]] = {}
        for idx, p in enumerate(self.points):
            proj = project_point(p, k_l, extrinsics)
            if proj is None:
                continue
            u, v, depth = proj
            if u < 0 or v < 0 or u >= w or v >= h:
                continue
            buckets.setdefault((u, v), []).append((depth, idx))
        for (u, v), items in buckets.items():
            _, best = min(items, key=lambda t: t[0])
            z_hat[:, v, u] = self.features[best]
            mask[v, u] = 1.0
        return z_hat, mask

    def footprint_bytes(self) -> int:
        """Approximate cache size: 3D coords + latent features."""
        if not self.points:
            return 0
        pts = np.stack(self.points)
        feats = np.stack(self.features)
        return int(pts.nbytes + feats.nbytes)


@dataclass
class RGBPointCloudMemory:
    """RGB baseline M_rgb = {(p_i, c_i)} for efficiency comparison (Eq. 2)."""

    points: list[np.ndarray] = field(default_factory=list)
    colors: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def footprint_bytes(self) -> int:
        if not self.points:
            return 0
        pts = np.stack(self.points)
        cols = np.stack(self.colors)
        return int(pts.nbytes + cols.nbytes)

    def estimated_readout_cost(
        self,
        *,
        rgb_hw: tuple[int, int],
        latent_hw: tuple[int, int],
        vae_encode_flops_scale: float = 1.0,
    ) -> dict[str, float]:
        """Model rasterise + VAE encode cost for one conditioning step."""
        H, W = rgb_hw
        h, w = latent_hw
        n = len(self.points)
        raster = n * np.log2(max(n, 2)) + H * W
        encode = H * W * vae_encode_flops_scale
        return {"n_points": float(n), "raster_cost": float(raster), "vae_encode_cost": float(encode), "total": float(raster + encode)}


def latent_readout_cost(n_points: int, latent_hw: tuple[int, int]) -> dict[str, float]:
    """Model latent-resolution projection cost (no VAE encode)."""
    h, w = latent_hw
    proj = n_points * np.log2(max(n_points, 2)) + h * w
    return {"n_points": float(n_points), "projection_cost": float(proj), "vae_encode_cost": 0.0, "total": float(proj)}
