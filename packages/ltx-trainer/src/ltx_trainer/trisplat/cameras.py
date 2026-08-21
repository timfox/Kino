"""Camera utilities for TriSplat (numpy, compatible with GOPEX SfM export)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TriSplatCamera:
    image_path: Path
    width: int
    height: int
    K: np.ndarray  # 3x3 intrinsics
    R_cw: np.ndarray  # 3x3 camera-to-world rotation
    t_cw: np.ndarray  # 3 camera-to-world translation
    view_id: int = 0

    @property
    def R_wc(self) -> np.ndarray:
        return self.R_cw.T

    @property
    def t_wc(self) -> np.ndarray:
        return -self.R_wc @ self.t_cw

    def world_to_camera(self, pts: np.ndarray) -> np.ndarray:
        """``pts`` shape ``(..., 3)`` in world frame."""
        return (self.R_wc @ pts[..., None]).squeeze(-1) + self.t_wc

    def camera_to_world(self, pts: np.ndarray) -> np.ndarray:
        return (self.R_cw @ pts[..., None]).squeeze(-1) + self.t_cw

    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = pts_cam[..., 2].clip(min=1e-6)
        u = self.K[0, 0] * pts_cam[..., 0] / z + self.K[0, 2]
        v = self.K[1, 1] * pts_cam[..., 1] / z + self.K[1, 2]
        return u, v


def load_cameras_from_sfm(sfm_dir: Path, images_root: Path | None = None) -> list[TriSplatCamera]:
    sfm_dir = Path(sfm_dir)
    cam_json = sfm_dir / "cameras.json"
    if not cam_json.is_file():
        raise FileNotFoundError(f"No cameras.json in {sfm_dir}")
    raw = json.loads(cam_json.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "frames" in raw:
        entries = raw["frames"]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"Unsupported cameras.json layout: {cam_json}")

    root = images_root or sfm_dir / "frames"
    if not root.is_dir():
        root = sfm_dir / "colmap_workspace" / "images"
    cams: list[TriSplatCamera] = []
    for i, entry in enumerate(entries):
        K = np.asarray(entry["K"], dtype=np.float64)
        R_wc = np.asarray(entry.get("R_wc") or entry.get("R"), dtype=np.float64)
        t_wc = np.asarray(entry.get("t_wc") or entry.get("t"), dtype=np.float64).reshape(3)
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        name = entry.get("image_name") or entry.get("image") or entry.get("file_path") or f"frame_{i:06d}.jpg"
        path = Path(name)
        if not path.is_file():
            path = root / Path(name).name
        if not path.is_file():
            path = root / name
        cams.append(
            TriSplatCamera(
                image_path=path,
                width=int(entry["width"]),
                height=int(entry["height"]),
                K=K,
                R_cw=R_cw,
                t_cw=t_cw,
                view_id=int(entry.get("frame_index", i)),
            )
        )
    return cams


def relative_first_view(cameras: list[TriSplatCamera]) -> list[TriSplatCamera]:
    """Express all poses relative to the first camera (paper gauge fixing)."""
    if not cameras:
        return cameras
    ref = cameras[0]
    R0 = ref.R_cw
    t0 = ref.t_cw
    R0_inv = R0.T
    out: list[TriSplatCamera] = []
    for c in cameras:
        R_rel = R0_inv @ c.R_cw
        t_rel = R0_inv @ (c.t_cw - t0)
        out.append(
            TriSplatCamera(
                image_path=c.image_path,
                width=c.width,
                height=c.height,
                K=c.K.copy(),
                R_cw=R_rel,
                t_cw=t_rel,
                view_id=c.view_id,
            )
        )
    return out
