"""Frame-wise SKILD super-resolution for video clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ltx_trainer.skild.config import SkildConfig
from ltx_trainer.skild.inference import SkildSpectrum, continuous_super_resolve


def super_resolve_video_frames(
    frames: list[np.ndarray],
    *,
    scale_factor: float,
    spectrum: SkildSpectrum,
    cfg: SkildConfig | None = None,
    seed: int = 0,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i, fr in enumerate(frames):
        out.append(
            continuous_super_resolve(
                fr,
                scale_factor=scale_factor,
                spectrum=spectrum,
                cfg=cfg,
                seed=seed + i,
            )
        )
    return out


def read_video_rgb(path: Path, *, max_frames: int | None = None) -> list[np.ndarray]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while cap.isOpened():
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(rgb)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return frames


def write_video_rgb(path: Path, frames: list[np.ndarray], *, fps: float = 24.0) -> None:
    import cv2  # noqa: PLC0415

    if not frames:
        raise ValueError("No frames")
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    try:
        for fr in frames:
            bgr = cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()
