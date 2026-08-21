"""End-to-end equirectangular 360° video photogrammetry."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ltx_trainer.pano360.ingest import Pano360Probe, probe_pano360_source, write_probe_json
from ltx_trainer.pano360.views import ViewSpec, default_ring_views, extract_views_from_equirect_frame

SfmBackend = Literal["auto", "colmap", "opencv"]


@dataclass
class Pano360PhotogrammetryConfig:
    output_dir: Path
    target_fps: float = 2.0
    max_frames: int = 60
    views_per_frame: int = 8
    view_fov_deg: float = 90.0
    view_width: int = 640
    view_height: int = 360
    extra_view_pitches: tuple[float, ...] = (-25.0, 25.0)
    sfm_backend: SfmBackend = "auto"
    require_equirect: bool = True


@dataclass
class Pano360PhotogrammetryResult:
    probe: Pano360Probe
    equirect_frames_dir: Path
    perspective_views_dir: Path
    sfm_dir: Path
    view_schedule: list[ViewSpec] = field(default_factory=list)
    num_equirect_frames: int = 0
    num_perspective_images: int = 0
    sfm_backend: str = "opencv"
    sfm_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe.to_dict(),
            "equirect_frames_dir": str(self.equirect_frames_dir),
            "perspective_views_dir": str(self.perspective_views_dir),
            "sfm_dir": str(self.sfm_dir),
            "view_schedule": [
                {"yaw_deg": v.yaw_deg, "pitch_deg": v.pitch_deg, "fov_deg": v.fov_deg, "name": v.label}
                for v in self.view_schedule
            ],
            "num_equirect_frames": self.num_equirect_frames,
            "num_perspective_images": self.num_perspective_images,
            "sfm_backend": self.sfm_backend,
            "sfm_summary": self.sfm_summary,
        }


def _extract_equirect_frames(
    probe: Pano360Probe,
    out_dir: Path,
    *,
    target_fps: float,
    max_frames: int,
) -> list[Path]:
    from ltx_trainer.ffmpeg_io import ffmpeg_bin, require_ffmpeg

    out_dir.mkdir(parents=True, exist_ok=True)
    if probe.kind == "equirect_still":
        import cv2  # noqa: PLC0415

        img = cv2.imread(str(probe.decode_path))
        if img is None:
            raise RuntimeError(f"Could not read {probe.decode_path}")
        dst = out_dir / "frame_0000.jpg"
        cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return [dst]

    require_ffmpeg()
    pattern = str(out_dir / "frame_%04d.jpg")
    vf = f"fps={target_fps}"
    cmd = [
        ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(probe.decode_path),
        "-vf",
        vf,
        "-frames:v",
        str(max_frames),
        "-q:v",
        "2",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames extracted from {probe.decode_path}")
    return frames


def _load_rgb(path: Path) -> Any:
    import cv2  # noqa: PLC0415

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_perspective_views(
    equirect_frames: list[Path],
    views_dir: Path,
    views: list[ViewSpec],
    *,
    out_width: int,
    out_height: int,
) -> list[Path]:
    views_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fi, frame_path in enumerate(equirect_frames):
        rgb = _load_rgb(frame_path)
        written.extend(
            extract_views_from_equirect_frame(
                rgb,
                views,
                views_dir,
                fi,
                out_width=out_width,
                out_height=out_height,
            )
        )
    return written


def run_pano360_photogrammetry(
    source: str | Path,
    config: Pano360PhotogrammetryConfig | None = None,
) -> Pano360PhotogrammetryResult:
    """Ingest equirect 360° video → perspective tiles → SfM export."""
    from ltx_trainer.sfm_export import export_sfm_from_images

    cfg = config or Pano360PhotogrammetryConfig(output_dir=Path("pano360_out"))
    out = Path(cfg.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    probe = probe_pano360_source(source)
    write_probe_json(probe, out / "pano360_probe.json")
    if cfg.require_equirect and not probe.is_equirectangular:
        raise ValueError(
            f"Source is not 2:1 equirectangular ({probe.width}x{probe.height}). "
            "Use Insta360 ingest with LTX_INSTA360_PROJECTION=equirect or export equirect MP4."
        )

    views = default_ring_views(
        cfg.views_per_frame,
        fov_deg=cfg.view_fov_deg,
        extra_pitches=cfg.extra_view_pitches,
    )

    equirect_dir = out / "equirect_frames"
    if equirect_dir.exists():
        shutil.rmtree(equirect_dir)
    equirect_frames = _extract_equirect_frames(
        probe,
        equirect_dir,
        target_fps=cfg.target_fps,
        max_frames=cfg.max_frames,
    )

    views_dir = out / "perspective_views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    persp_paths = extract_perspective_views(
        equirect_frames,
        views_dir,
        views,
        out_width=cfg.view_width,
        out_height=cfg.view_height,
    )

    sfm_dir = out / "sfm"
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    sfm_result = export_sfm_from_images(
        views_dir,
        sfm_dir,
        backend=cfg.sfm_backend,
        source_video=str(probe.source_path),
        pano360_meta={
            "projection": "equirectangular",
            "num_equirect_frames": len(equirect_frames),
            "views_per_frame": len(views),
            "view_schedule": [v.label for v in views],
        },
    )

    summary_path = sfm_dir / "sfm_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    result = Pano360PhotogrammetryResult(
        probe=probe,
        equirect_frames_dir=equirect_dir,
        perspective_views_dir=views_dir,
        sfm_dir=sfm_dir,
        view_schedule=views,
        num_equirect_frames=len(equirect_frames),
        num_perspective_images=len(persp_paths),
        sfm_backend=sfm_result.backend,
        sfm_summary=summary,
    )
    (out / "pano360_result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
