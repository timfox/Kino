"""Multi-view camera loading for P2GS (SfM / GOPEX ``cameras.json``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass
class Camera:
    """Pinhole camera with world-to-camera pose."""

    image_path: Path
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: Tensor  # [3, 3] world-to-camera rotation
    t: Tensor  # [3] world-to-camera translation

    @property
    def center(self) -> Tensor:
        """Camera center in world coordinates."""
        return -(self.R.T @ self.t)

    def world_to_cam(self, points: Tensor) -> Tensor:
        """``points`` [N, 3] world → camera [N, 3]."""
        return (points @ self.R.T) + self.t.unsqueeze(0)

    def project(self, points_cam: Tensor) -> tuple[Tensor, Tensor]:
        """Project camera-space points to pixel coords and depth."""
        z = points_cam[:, 2].clamp(min=1e-4)
        u = self.fx * (points_cam[:, 0] / z) + self.cx
        v = self.fy * (points_cam[:, 1] / z) + self.cy
        return torch.stack([u, v], dim=-1), z


def _c2w_to_w2c(transform: list[list[float]]) -> tuple[Tensor, Tensor]:
    c2w = torch.tensor(transform, dtype=torch.float32)
    w2c = torch.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return R, t


def load_scene_cameras(scene_dir: str | Path) -> list[Camera]:
    """Load ``cameras.json`` frames from an SfM export directory."""
    root = Path(scene_dir).expanduser().resolve()
    cam_json = root / "cameras.json"
    if not cam_json.is_file():
        raise FileNotFoundError(f"Missing {cam_json}")
    data = json.loads(cam_json.read_text(encoding="utf-8"))
    frames = data.get("frames", data if isinstance(data, list) else [])
    cameras: list[Camera] = []
    for fr in frames:
        rel = fr["file_path"]
        img_path = (root / rel).resolve()
        if not img_path.is_file():
            alt = root / "images" / Path(rel).name
            img_path = alt if alt.is_file() else img_path
        R, t = _c2w_to_w2c(fr["transform_matrix"])
        cameras.append(
            Camera(
                image_path=img_path,
                width=int(fr["width"]),
                height=int(fr["height"]),
                fx=float(fr["fx"]),
                fy=float(fr["fy"]),
                cx=float(fr["cx"]),
                cy=float(fr["cy"]),
                R=R,
                t=t,
            )
        )
    if not cameras:
        raise ValueError(f"No frames in {cam_json}")
    return cameras


def load_image_chw(cam: Camera, *, scale: float = 1.0, device: torch.device | str = "cpu") -> Tensor:
    """Load LDR ground truth as ``[3, H, W]`` float in ``[0, 1]`` (sRGB)."""
    from PIL import Image

    img = Image.open(cam.image_path).convert("RGB")
    if scale != 1.0:
        w = max(1, int(cam.width * scale))
        h = max(1, int(cam.height * scale))
        img = img.resize((w, h), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def init_points_from_ply(scene_dir: Path, max_points: int = 10_000) -> Tensor | None:
    """Load COLMAP-style ``sparse.ply`` XYZ if present."""
    ply = scene_dir / "sparse.ply"
    if not ply.is_file():
        return None
    pts: list[list[float]] = []
    with ply.open("r", encoding="utf-8", errors="ignore") as f:
        in_vertex = False
        for line in f:
            if line.startswith("element vertex"):
                in_vertex = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            if line.startswith("end_header"):
                continue
            if in_vertex:
                parts = line.split()
                if len(parts) >= 3:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not pts:
        return None
    t = torch.tensor(pts, dtype=torch.float32)
    if t.shape[0] > max_points:
        idx = torch.randperm(t.shape[0])[:max_points]
        t = t[idx]
    return t
