"""Validation sample sharpness checks (Laplacian variance proxy)."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleQualityReport:
    path: Path
    laplacian_var: float
    mean_luma: float
    file_bytes: int
    ok: bool
    reason: str = ""


def _frame_laplacian_var(image_path: Path) -> tuple[float, float]:
    try:
        import cv2  # type: ignore[import-untyped]
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python required for validation quality checks") from exc

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not decode frame from {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return lap, float(np.mean(gray))


def score_video_sample(
    path: Path | str,
    *,
    min_laplacian_var: float = 80.0,
    min_file_bytes: int = 100_000,
    min_mean_luma: float = 5.0,
) -> SampleQualityReport:
    """Score one validation MP4 (first frame). Returns report; ``ok`` False if likely garbage."""
    p = Path(path)
    size = p.stat().st_size if p.is_file() else 0
    if size < min_file_bytes:
        return SampleQualityReport(
            path=p,
            laplacian_var=0.0,
            mean_luma=0.0,
            file_bytes=size,
            ok=False,
            reason=f"file too small ({size} B < {min_file_bytes})",
        )
    with tempfile.TemporaryDirectory(prefix="ltx-val-q-") as td:
        frame = Path(td) / "frame.png"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-vframes", "1", str(frame)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not frame.is_file():
            return SampleQualityReport(
                path=p,
                laplacian_var=0.0,
                mean_luma=0.0,
                file_bytes=size,
                ok=False,
                reason="ffmpeg frame extract failed",
            )
        lap, mean_luma = _frame_laplacian_var(frame)
    ok = lap >= min_laplacian_var and mean_luma >= min_mean_luma
    reason = ""
    if not ok:
        if mean_luma < min_mean_luma:
            reason = f"mean luma {mean_luma:.1f} < {min_mean_luma}"
        else:
            reason = f"laplacian var {lap:.1f} < {min_laplacian_var} (color noise / blur)"
    return SampleQualityReport(
        path=p,
        laplacian_var=lap,
        mean_luma=mean_luma,
        file_bytes=size,
        ok=ok,
        reason=reason,
    )


def check_sample_paths(
    paths: list[Path | str],
    *,
    min_laplacian_var: float = 80.0,
) -> list[SampleQualityReport]:
    reports = [score_video_sample(p, min_laplacian_var=min_laplacian_var) for p in paths]
    return reports


def assert_samples_ok(
    paths: list[Path | str],
    *,
    min_laplacian_var: float = 80.0,
) -> list[SampleQualityReport]:
    reports = check_sample_paths(paths, min_laplacian_var=min_laplacian_var)
    bad = [r for r in reports if not r.ok]
    if bad:
        lines = [f"{r.path.name}: {r.reason} (lap={r.laplacian_var:.1f}, luma={r.mean_luma:.1f})" for r in bad]
        raise RuntimeError("Validation sample quality gate failed:\n  " + "\n  ".join(lines))
    return reports
