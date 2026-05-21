"""Insta360 container ingest (``.insv`` / ``.insp``) for LTX dataset prep.

Insta360 cameras write proprietary containers that usually wrap **H.264 / HEVC** bitstreams.
Some files expose **two video streams** (dual fisheye); others use one very wide frame with
side-by-side circles. This module:

1. **Probes** all video streams via ffprobe.
2. **Builds** an ffmpeg ``-filter_complex`` (``hstack`` + ``v360`` when needed).
3. **Optionally transcodes** to a cached flat MP4 proxy for PyAV / downstream tools.

Environment:

- ``LTX_INSTA360_DECODE`` — ``proxy`` (default): cache stitched MP4 under ``.gopex_insta360_cache/``
- ``LTX_INSTA360_DECODE`` — ``direct``: pass ``.insv`` to ffmpeg decode with inline filters
- ``LTX_INSTA360_DECODE`` — ``off``: do not special-case (may fail on raw ``.insv``)
- ``LTX_INSTA360_PROJECTION`` — ``equirect`` | ``flat`` | ``dfisheye`` (default ``equirect``)
- ``LTX_INSTA360_OUT_WIDTH`` / ``LTX_INSTA360_OUT_HEIGHT`` — stitched output size (default 1920×960)
- ``LTX_INSTA360_CACHE_DIR`` — override cache root (default: sibling ``.gopex_insta360_cache``)

Official Insta360 Studio export to MP4 remains the most compatible path; this module automates
the common ffmpeg stitching route for batch pipelines.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ltx_trainer.media_formats import (
    INSTA360_STILL_SUFFIXES,
    INSTA360_VIDEO_SUFFIXES,
    is_insta360_path,
    is_insta360_still_path,
    is_insta360_video_path,
)

Insta360Projection = Literal["equirect", "flat", "dfisheye"]
Insta360Layout = Literal["dual_stream", "wide_single", "single", "unknown"]


class Insta360IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Insta360VideoStream:
    index: int
    codec_name: str
    width: int
    height: int
    avg_fps: float
    nb_frames: int | None = None
    pix_fmt: str | None = None


@dataclass
class Insta360Probe:
    path: Path
    format_name: str
    duration_sec: float
    video_streams: list[Insta360VideoStream] = field(default_factory=list)
    audio_codec: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    layout: Insta360Layout = "unknown"
    wide_aspect: float | None = None

    @property
    def primary(self) -> Insta360VideoStream | None:
        return self.video_streams[0] if self.video_streams else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def insta360_decode_mode() -> str:
    return os.environ.get("LTX_INSTA360_DECODE", "proxy").strip().lower()


def insta360_projection() -> Insta360Projection:
    raw = os.environ.get("LTX_INSTA360_PROJECTION", "equirect").strip().lower()
    if raw in ("equirect", "flat", "dfisheye"):
        return raw  # type: ignore[return-value]
    return "equirect"


def insta360_output_size() -> tuple[int, int]:
    w = int(os.environ.get("LTX_INSTA360_OUT_WIDTH", "1920"))
    h = int(os.environ.get("LTX_INSTA360_OUT_HEIGHT", "960"))
    return max(320, w - w % 2), max(180, h - h % 2)


def _cache_dir_for(source: Path) -> Path:
    env = os.environ.get("LTX_INSTA360_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return source.parent / ".gopex_insta360_cache"


def _cache_key(source: Path, projection: str, out_w: int, out_h: int) -> str:
    st = source.stat()
    blob = f"{source.resolve()}|{st.st_mtime_ns}|{st.st_size}|{projection}|{out_w}x{out_h}"
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def probe_insta360(path: str | Path) -> Insta360Probe:
    """Probe Insta360 media; requires ffprobe."""
    from ltx_trainer.ffmpeg_io import _parse_fps, ffprobe_json, require_ffmpeg

    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise Insta360IngestError(f"Not a file: {p}")
    require_ffmpeg()
    raw = ffprobe_json(p)
    fmt = raw.get("format") or {}
    duration = float(fmt.get("duration") or 0.01)
    tags = {str(k): str(v) for k, v in (fmt.get("tags") or {}).items()}

    streams: list[Insta360VideoStream] = []
    audio_codec: str | None = None
    for st in raw.get("streams") or []:
        if not isinstance(st, dict):
            continue
        if st.get("codec_type") == "audio" and audio_codec is None:
            audio_codec = str(st.get("codec_name") or "")
        if st.get("codec_type") != "video":
            continue
        nb = st.get("nb_frames")
        streams.append(
            Insta360VideoStream(
                index=int(st.get("index") or 0),
                codec_name=str(st.get("codec_name") or "unknown"),
                width=int(st.get("width") or 0),
                height=int(st.get("height") or 0),
                avg_fps=_parse_fps(st.get("avg_frame_rate") or st.get("r_frame_rate")),
                nb_frames=int(nb) if nb not in (None, "N/A") else None,
                pix_fmt=str(st.get("pix_fmt")) if st.get("pix_fmt") else None,
            )
        )

    layout: Insta360Layout = "unknown"
    wide_aspect: float | None = None
    if len(streams) >= 2:
        layout = "dual_stream"
    elif len(streams) == 1:
        s0 = streams[0]
        if s0.width > 0 and s0.height > 0:
            wide_aspect = s0.width / s0.height
            if wide_aspect >= 1.85:
                layout = "wide_single"
            else:
                layout = "single"

    return Insta360Probe(
        path=p,
        format_name=str(fmt.get("format_name") or "unknown"),
        duration_sec=max(0.01, duration),
        video_streams=streams,
        audio_codec=audio_codec,
        tags=tags,
        layout=layout,
        wide_aspect=wide_aspect,
    )


def build_insta360_video_filter(
    probe: Insta360Probe,
    *,
    projection: Insta360Projection | None = None,
    out_width: int | None = None,
    out_height: int | None = None,
) -> tuple[str, list[str]]:
    """Return ``(filter_graph, extra_input_args)`` for ffmpeg.

    ``filter_graph`` may be a ``-vf`` chain (single stream) or ``-filter_complex`` graph.
    ``extra_input_args`` is usually empty; reserved for future multi-file inputs.
    """
    proj = projection or insta360_projection()
    ow, oh = out_width or insta360_output_size()[0], out_height or insta360_output_size()[1]
    scale_even = f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"

    if proj == "flat":
        if probe.layout == "dual_stream" and len(probe.video_streams) >= 2:
            return f"[0:v:0][0:v:1]hstack=inputs=2,{scale_even}[vout]", ["-map", "[vout]"]
        return scale_even, []

    # dfisheye / equirect stitching via ffmpeg v360 (requires libavfilter v360)
    out_mode = "equirect" if proj == "equirect" else "dfisheye"
    v360 = (
        f"v360=input=dfisheye:ih_fov=210:iv_fov=210:output={out_mode}:w={ow}:h={oh}:"
        "interp=lanczos:reset_rot=1"
    )

    if probe.layout == "dual_stream" and len(probe.video_streams) >= 2:
        graph = f"[0:v:0][0:v:1]hstack=inputs=2,{v360}[vout]"
        return graph, ["-map", "[vout]"]

    if probe.layout == "wide_single":
        graph = f"[0:v:0]{v360}[vout]"
        return graph, ["-map", "[vout]"]

    if probe.layout == "single":
        # Likely already rectilinear export inside INSV
        return scale_even, []

    return scale_even, []


def transcode_insta360_proxy(
    path: str | Path,
    *,
    dst: Path | None = None,
    projection: Insta360Projection | None = None,
    max_frames: int | None = None,
    fps: float | None = None,
    overwrite: bool = False,
) -> Path:
    """Transcode Insta360 video to a flat H.264 MP4 proxy."""
    from ltx_trainer.ffmpeg_io import FFmpegError, _run, ffmpeg_bin, hwaccel_input_args, require_ffmpeg

    source = Path(path).expanduser().resolve()
    if not is_insta360_video_path(source):
        raise Insta360IngestError(f"Not an Insta360 video suffix: {source}")

    probe = probe_insta360(source)
    if not probe.video_streams:
        raise Insta360IngestError(f"No video streams in {source}")

    proj = projection or insta360_projection()
    ow, oh = insta360_output_size()
    cache = _cache_dir_for(source)
    cache.mkdir(parents=True, exist_ok=True)
    if dst is None:
        dst = cache / f"{source.stem}_{_cache_key(source, proj, ow, oh)}.proxy.mp4"
    dst = Path(dst).expanduser().resolve()

    if dst.is_file() and not overwrite:
        return dst

    filter_graph, map_args = build_insta360_video_filter(probe, projection=proj, out_width=ow, out_height=oh)
    ff = ffmpeg_bin()
    require_ffmpeg()
    loglevel = os.environ.get("LTX_FFMPEG_LOGLEVEL", "error")

    cmd: list[str] = [ff, "-hide_banner", "-loglevel", loglevel, "-y", *hwaccel_input_args(), "-i", str(source)]
    use_complex = filter_graph.startswith("[") or "[vout]" in filter_graph
    if use_complex:
        cmd += ["-filter_complex", filter_graph, *map_args]
    else:
        cmd += ["-vf", filter_graph]
        cmd += ["-map", "0:v:0"]

    if fps is not None and fps > 0:
        cmd += ["-r", str(fps)]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]

    cmd += [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        os.environ.get("LTX_INSTA360_PRESET", "veryfast"),
        "-crf",
        os.environ.get("LTX_INSTA360_CRF", "20"),
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]

    try:
        _run(cmd)
    except FFmpegError as exc:
        raise Insta360IngestError(
            f"Insta360 transcode failed for {source}. "
            "Try exporting MP4 from Insta360 Studio, or set LTX_INSTA360_PROJECTION=flat. "
            f"ffmpeg error: {exc}"
        ) from exc

    if not dst.is_file() or dst.stat().st_size == 0:
        raise Insta360IngestError(f"Proxy transcode produced no output: {dst}")
    return dst


def extract_insta360_still(path: str | Path, *, dst_jpeg: Path | None = None) -> Path:
    """Extract first frame from ``.insp`` (or video-like INSV) to JPEG."""
    from ltx_trainer.ffmpeg_io import extract_frame_jpeg, require_ffmpeg

    source = Path(path).expanduser().resolve()
    if not is_insta360_still_path(source) and not is_insta360_video_path(source):
        raise Insta360IngestError(f"Not Insta360 still/video: {source}")
    require_ffmpeg()
    out = dst_jpeg or (_cache_dir_for(source) / f"{source.stem}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    if is_insta360_video_path(source):
        proxy = transcode_insta360_proxy(source, max_frames=1, fps=1.0)
        if not extract_frame_jpeg(proxy, 0.0, out):
            raise Insta360IngestError(f"Failed to extract frame from {proxy}")
    elif not extract_frame_jpeg(source, 0.0, out):
        raise Insta360IngestError(f"Failed to extract still from {source}")
    return out


def resolve_insta360_decode_path(path: str | Path) -> Path:
    """Return a path suitable for ``video_utils`` / ``ffmpeg_io`` decode.

    - Non-Insta360 paths are returned unchanged.
    - ``LTX_INSTA360_DECODE=off`` returns the original path.
    - ``proxy`` returns cached stitched MP4 (default).
    - ``direct`` returns original (caller must pass filters — see ``read_rgb_frames``).
    """
    p = Path(path).expanduser().resolve()
    if not is_insta360_path(p):
        return p
    mode = insta360_decode_mode()
    if mode in ("0", "false", "no", "off"):
        return p
    if is_insta360_still_path(p) and not is_insta360_video_path(p):
        return extract_insta360_still(p)
    if mode == "direct":
        return p
    return transcode_insta360_proxy(p)


def ingest_insta360_for_ltx(
    path: str | Path,
    *,
    projection: Insta360Projection | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """High-level ingest: probe + optional proxy + metadata for manifests.

    Returns a dict suitable for JSON logging or dataset sidecars.
    """
    source = Path(path).expanduser().resolve()
    probe = probe_insta360(source) if is_insta360_video_path(source) else None
    decode_path = source
    if is_insta360_video_path(source) and insta360_decode_mode() not in ("0", "false", "no", "off"):
        decode_path = transcode_insta360_proxy(
            source,
            projection=projection,
            overwrite=force_refresh,
        )
    elif is_insta360_still_path(source):
        decode_path = extract_insta360_still(source)

    return {
        "source_path": str(source),
        "decode_path": str(decode_path),
        "insta360": True,
        "probe": probe.to_dict() if probe else None,
        "projection": projection or insta360_projection(),
        "decode_mode": insta360_decode_mode(),
    }


__all__ = [
    "Insta360IngestError",
    "Insta360Layout",
    "Insta360Probe",
    "Insta360Projection",
    "Insta360VideoStream",
    "build_insta360_video_filter",
    "extract_insta360_still",
    "ingest_insta360_for_ltx",
    "insta360_decode_mode",
    "insta360_projection",
    "probe_insta360",
    "resolve_insta360_decode_path",
    "transcode_insta360_proxy",
]
