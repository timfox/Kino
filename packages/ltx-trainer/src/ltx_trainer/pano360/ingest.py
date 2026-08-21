"""Resolve 360° sources (MP4 equirect, Insta360, still panoramas)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ltx_trainer.media_formats import is_insta360_video_path, is_video_path
from ltx_trainer.pano360.geometry import is_equirectangular_size

PanoSourceKind = Literal["equirect_video", "insta360", "equirect_still", "unknown"]


@dataclass
class Pano360Probe:
    source_path: Path
    decode_path: Path
    kind: PanoSourceKind
    width: int
    height: int
    duration_sec: float | None
    fps: float | None
    is_equirectangular: bool
    projection: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_path"] = str(self.source_path)
        d["decode_path"] = str(self.decode_path)
        return d


def probe_pano360_source(path: str | Path) -> Pano360Probe:
    """Probe file; for Insta360 returns stitched equirect proxy dimensions when possible."""
    from ltx_trainer.ffmpeg_io import ffprobe_json, require_ffmpeg

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    decode = source
    kind: PanoSourceKind = "unknown"
    projection = "equirectangular"

    if is_insta360_video_path(source):
        from ltx_trainer.insta360_ingest import ingest_insta360_for_ltx

        meta = ingest_insta360_for_ltx(source, projection="equirect")
        decode = Path(meta["decode_path"])
        kind = "insta360"
        projection = str(meta.get("projection") or "equirect")
    elif source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        kind = "equirect_still"
    elif is_video_path(source):
        kind = "equirect_video"

    require_ffmpeg()
    raw = ffprobe_json(decode)
    fmt = raw.get("format") or {}
    duration = float(fmt.get("duration") or 0.0) if kind != "equirect_still" else None
    width = height = 0
    fps: float | None = None
    for st in raw.get("streams") or []:
        if st.get("codec_type") != "video":
            continue
        width = int(st.get("width") or 0)
        height = int(st.get("height") or 0)
        rfr = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
        if isinstance(rfr, str) and "/" in rfr:
            num, den = rfr.split("/", 1)
            fps = float(num) / max(float(den), 1e-6)
        break

    if kind == "equirect_still":
        import cv2  # noqa: PLC0415

        img = cv2.imread(str(decode))
        if img is None:
            raise RuntimeError(f"Could not read still: {decode}")
        height, width = img.shape[:2]

    eq = is_equirectangular_size(width, height)
    return Pano360Probe(
        source_path=source,
        decode_path=decode,
        kind=kind,
        width=width,
        height=height,
        duration_sec=duration,
        fps=fps,
        is_equirectangular=eq,
        projection=projection,
    )


def write_probe_json(probe: Pano360Probe, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(probe.to_dict(), indent=2), encoding="utf-8")
