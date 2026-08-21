"""Perspective view extraction from equirectangular frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ltx_trainer.pano360.geometry import build_equirect_remap


@dataclass(frozen=True)
class ViewSpec:
    yaw_deg: float
    pitch_deg: float
    fov_deg: float = 90.0
    name: str | None = None

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        return f"y{self.yaw_deg:.0f}_p{self.pitch_deg:.0f}"


def default_ring_views(
    count: int = 8,
    *,
    pitch_deg: float = 0.0,
    fov_deg: float = 90.0,
    extra_pitches: tuple[float, ...] = (),
) -> list[ViewSpec]:
    """Horizontal ring plus optional pitched views (e.g. ±25°)."""
    views = [
        ViewSpec(yaw_deg=i * 360.0 / count, pitch_deg=pitch_deg, fov_deg=fov_deg, name=f"ring_{i:02d}")
        for i in range(max(4, count))
    ]
    for p in extra_pitches:
        views.append(ViewSpec(yaw_deg=0.0, pitch_deg=p, fov_deg=fov_deg, name=f"pitch_{p:+.0f}"))
    return views


def extract_pinhole_view(
    equirect_rgb: np.ndarray,
    spec: ViewSpec,
    *,
    out_width: int = 640,
    out_height: int = 360,
) -> np.ndarray:
    """Sample a pinhole view from an equirect RGB uint8 array ``(H, W, 3)``."""
    import cv2  # noqa: PLC0415

    eh, ew = equirect_rgb.shape[:2]
    map_x, map_y = build_equirect_remap(
        (eh, ew),
        (out_height, out_width),
        yaw_deg=spec.yaw_deg,
        pitch_deg=spec.pitch_deg,
        fov_deg=spec.fov_deg,
    )
    return cv2.remap(
        equirect_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def write_view_jpeg(path: Path, rgb: np.ndarray, *, quality: int = 92) -> None:
    import cv2  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def extract_views_from_equirect_frame(
    equirect_rgb: np.ndarray,
    views: list[ViewSpec],
    out_dir: Path,
    frame_index: int,
    *,
    out_width: int,
    out_height: int,
) -> list[Path]:
    """Write ``frame_XXXX_view_YY.jpg`` files; return paths."""
    paths: list[Path] = []
    for vi, spec in enumerate(views):
        rgb = extract_pinhole_view(
            equirect_rgb,
            spec,
            out_width=out_width,
            out_height=out_height,
        )
        name = f"frame_{frame_index:04d}_{spec.label}.jpg"
        dst = out_dir / name
        write_view_jpeg(dst, rgb)
        paths.append(dst)
    return paths
