"""FFmpeg / ffprobe integration for LTX video ingest, dataset prep, and tooling.

Centralizes subprocess calls that were scattered across ``tools/`` scripts. PyAV remains the default
decode path in :mod:`ltx_trainer.video_utils`; set ``LTX_VIDEO_BACKEND=ffmpeg`` to decode via
``ffmpeg`` rawvideo pipes (useful for filter graphs, hardware decode, and consistent logging).

Environment:

- ``LTX_FFMPEG_BIN`` / ``LTX_FFPROBE_BIN`` — binary paths (default: ``ffmpeg`` / ``ffprobe`` on ``PATH``)
- ``LTX_VIDEO_BACKEND`` — ``pyav`` (default) or ``ffmpeg`` for :func:`video_utils.read_video`
- ``LTX_FFMPEG_HWACCEL`` — ``none`` | ``auto`` | ``cuda`` | ``vaapi`` (prepended before ``-i`` when set)
- ``LTX_FFMPEG_LOGLEVEL`` — ffmpeg ``-loglevel`` (default ``error``)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

_PTS_TIME = re.compile(r"pts_time:([\d.]+)")


class FFmpegError(RuntimeError):
    """``ffmpeg`` / ``ffprobe`` failed."""

    def __init__(self, message: str, *, cmd: list[str] | None = None, stderr: str = "") -> None:
        self.cmd = cmd
        self.stderr = stderr
        extra = f"\nstderr:\n{stderr[-4000:]}" if stderr else ""
        super().__init__(f"{message}{extra}")


def ffmpeg_bin() -> str:
    return os.environ.get("LTX_FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"


def ffprobe_bin() -> str:
    return os.environ.get("LTX_FFPROBE_BIN", "ffprobe").strip() or "ffprobe"


def require_ffmpeg() -> tuple[str, str]:
    """Return ``(ffmpeg, ffprobe)`` paths or raise if missing."""
    ff = shutil.which(ffmpeg_bin())
    fp = shutil.which(ffprobe_bin())
    if not ff or not fp:
        raise FFmpegError(
            "ffmpeg and ffprobe must be on PATH (or set LTX_FFMPEG_BIN / LTX_FFPROBE_BIN). "
            f"Resolved ffmpeg={ff!r} ffprobe={fp!r}"
        )
    return ff, fp


def ffmpeg_version() -> str:
    ff, _ = require_ffmpeg()
    out = subprocess.run([ff, "-version"], check=True, capture_output=True, text=True)
    return (out.stdout or "").splitlines()[0]


def list_encoders(*, decoder: bool = True, encoder: bool = True) -> list[str]:
    """Parse ``ffmpeg -encoders`` / ``-decoders`` names (one pass)."""
    ff, _ = require_ffmpeg()
    names: list[str] = []
    for flag in ("-decoders", "-encoders"):
        if flag == "-decoders" and not decoder:
            continue
        if flag == "-encoders" and not encoder:
            continue
        out = subprocess.run([ff, "-hide_banner", flag], check=True, capture_output=True, text=True)
        for line in (out.stdout or "").splitlines():
            m = re.match(r"^\s*[VAS][\w.]*\s+(\S+)", line)
            if m:
                names.append(m.group(1))
    return sorted(set(names))


def hwaccel_input_args() -> list[str]:
    mode = os.environ.get("LTX_FFMPEG_HWACCEL", "none").strip().lower()
    if mode in ("", "none", "0", "false"):
        return []
    if mode == "auto":
        enc = list_encoders(decoder=True, encoder=False)
        if "h264_cuvid" in enc or "hevc_cuvid" in enc:
            return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        return []
    return ["-hwaccel", mode]


def _run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    text: bool = True,
    timeout_sec: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=text,
            timeout=timeout_sec,
        )
    except subprocess.CalledProcessError as e:
        raise FFmpegError(
            f"Command failed (exit {e.returncode}): {' '.join(cmd)}",
            cmd=cmd,
            stderr=(e.stderr or "") if text else "",
        ) from e
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"Command timed out after {timeout_sec}s: {' '.join(cmd)}", cmd=cmd) from e


def ffprobe_json(path: str | Path, *, select_streams: str | None = None) -> dict[str, Any]:
    """Run ``ffprobe -print_format json -show_format -show_streams``."""
    _, fp = require_ffmpeg()
    cmd = [
        fp,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
    ]
    if select_streams:
        cmd += ["-select_streams", select_streams]
    cmd.append(str(path))
    proc = _run(cmd)
    return json.loads(proc.stdout or "{}")


@dataclass(frozen=True)
class VideoStreamInfo:
    index: int
    codec_name: str
    width: int
    height: int
    avg_fps: float
    duration_sec: float | None
    nb_frames: int | None
    pix_fmt: str | None
    color_transfer: str | None
    color_primaries: str | None
    color_space: str | None
    color_range: str | None
    bit_rate: int | None


@dataclass(frozen=True)
class MediaProbe:
    path: Path
    format_name: str
    duration_sec: float
    video: VideoStreamInfo | None
    audio_codec: str | None
    tags: dict[str, str]

    @property
    def has_video(self) -> bool:
        return self.video is not None


def _parse_fps(rate: str | None) -> float:
    if not rate or rate in ("0/0", "N/A"):
        return 24.0
    try:
        if "/" in rate:
            return float(Fraction(rate))
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return 24.0


def probe_media(path: str | Path) -> MediaProbe:
    """Structured probe for the first video stream + container duration."""
    p = Path(path).expanduser().resolve()
    raw = ffprobe_json(p)
    fmt = raw.get("format") or {}
    duration = float(fmt.get("duration") or 0.0)
    tags = {str(k): str(v) for k, v in (fmt.get("tags") or {}).items()}

    video_info: VideoStreamInfo | None = None
    audio_codec: str | None = None
    for st in raw.get("streams") or []:
        if not isinstance(st, dict):
            continue
        ctype = st.get("codec_type")
        if ctype == "audio" and audio_codec is None:
            audio_codec = str(st.get("codec_name") or "")
        if ctype != "video" or video_info is not None:
            continue
        nb = st.get("nb_frames")
        video_info = VideoStreamInfo(
            index=int(st.get("index") or 0),
            codec_name=str(st.get("codec_name") or "unknown"),
            width=int(st.get("width") or 0),
            height=int(st.get("height") or 0),
            avg_fps=_parse_fps(st.get("avg_frame_rate") or st.get("r_frame_rate")),
            duration_sec=float(st["duration"]) if st.get("duration") else duration or None,
            nb_frames=int(nb) if nb not in (None, "N/A") else None,
            pix_fmt=str(st.get("pix_fmt")) if st.get("pix_fmt") else None,
            color_transfer=str(st.get("color_transfer")) if st.get("color_transfer") else None,
            color_primaries=str(st.get("color_primaries")) if st.get("color_primaries") else None,
            color_space=str(st.get("color_space")) if st.get("color_space") else None,
            color_range=str(st.get("color_range")) if st.get("color_range") else None,
            bit_rate=int(st["bit_rate"]) if st.get("bit_rate") not in (None, "N/A") else None,
        )
    return MediaProbe(
        path=p,
        format_name=str(fmt.get("format_name") or "unknown"),
        duration_sec=max(0.01, duration),
        video=video_info,
        audio_codec=audio_codec,
        tags=tags,
    )


def duration_seconds(path: str | Path) -> float:
    return probe_media(path).duration_sec


def estimate_frame_count(path: str | Path) -> int:
    """Fast frame count: ``nb_frames`` → ``duration * fps`` → decode count (last resort)."""
    pr = probe_media(path)
    if pr.video and pr.video.nb_frames and pr.video.nb_frames > 0:
        return int(pr.video.nb_frames)
    if pr.video and pr.video.duration_sec and pr.video.avg_fps:
        return max(1, int(round(pr.video.duration_sec * pr.video.avg_fps)))
    if pr.video and pr.duration_sec and pr.video.avg_fps:
        return max(1, int(round(pr.duration_sec * pr.video.avg_fps)))
  # fallback: ffmpeg frame count
    ff, _ = require_ffmpeg()
    cmd = [
        ff,
        "-hide_banner",
        "-loglevel",
        os.environ.get("LTX_FFMPEG_LOGLEVEL", "error"),
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"frame=\s*(\d+)", (proc.stderr or "") + (proc.stdout or ""))
    if m:
        return int(m.group(1))
    return max(1, int(round(pr.duration_sec * (pr.video.avg_fps if pr.video else 24.0))))


def build_scale_filter(*, max_width: int | None = None, max_height: int | None = None, fps: float | None = None) -> str:
    """Comma-separated ``-vf`` chain (even dimensions for yuv420)."""
    parts: list[str] = []
    if max_width or max_height:
        w = max_width or -1
        h = max_height or -1
        parts.append(f"scale='min({w},iw)':'min({h},ih)':force_original_aspect_ratio=decrease")
        parts.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    if fps is not None and fps > 0:
        parts.append(f"fps={fps}")
    return ",".join(parts) if parts else "scale=trunc(iw/2)*2:trunc(ih/2)*2"


def cut_segment(
    src: Path,
    start_sec: float,
    duration_sec: float,
    dst: Path,
    *,
    no_audio: bool = False,
    accurate_seek: bool = False,
    video_codec: str = "libx264",
    crf: int = 20,
    preset: str = "veryfast",
    pix_fmt: str = "yuv420p",
    audio_bitrate: str = "128k",
) -> bool:
    """Extract ``[start_sec, start_sec + duration)`` to ``dst`` (re-encode)."""
    ff, _ = require_ffmpeg()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    loglevel = os.environ.get("LTX_FFMPEG_LOGLEVEL", "error")
    before = [ff, "-hide_banner", "-loglevel", loglevel, "-y", *hwaccel_input_args()]
    if not accurate_seek:
        before += ["-ss", f"{start_sec:.3f}", "-i", str(src)]
        time_args: list[str] = ["-t", f"{duration_sec:.3f}"]
    else:
        before += ["-i", str(src), "-ss", f"{start_sec:.3f}"]
        time_args = ["-t", f"{duration_sec:.3f}"]
    vid = ["-c:v", video_codec, "-preset", preset, "-crf", str(crf), "-pix_fmt", pix_fmt]
    aud = ["-an"] if no_audio else ["-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2"]
    cmd = before + time_args + vid + aud + [str(dst)]
    try:
        _run(cmd)
    except FFmpegError:
        return False
    return dst.is_file() and dst.stat().st_size > 0


def extract_frame_jpeg(
    video: Path,
    t_sec: float,
    out_jpeg: Path,
    *,
    max_width: int = 896,
    qv: int = 2,
) -> bool:
    """Single-frame JPEG extract at ``t_sec``."""
    ff, _ = require_ffmpeg()
    out_jpeg = Path(out_jpeg)
    out_jpeg.parent.mkdir(parents=True, exist_ok=True)
    vf = build_scale_filter(max_width=max_width)
    cmd = [
        ff,
        "-hide_banner",
        "-loglevel",
        os.environ.get("LTX_FFMPEG_LOGLEVEL", "error"),
        "-y",
        *hwaccel_input_args(),
        "-ss",
        f"{t_sec:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        vf,
        "-q:v",
        str(qv),
        str(out_jpeg),
    ]
    try:
        _run(cmd)
    except FFmpegError:
        return False
    return out_jpeg.is_file() and out_jpeg.stat().st_size > 0


def sample_frame_times(duration: float | None, n: int) -> list[float]:
    if duration is None or duration <= 0:
        return [1.0 + i * 4.0 for i in range(n)]
    lo, hi = duration * 0.05, duration * 0.95
    if hi <= lo + 0.5:
        return [duration * 0.5]
    step = (hi - lo) / max(1, n - 1)
    return [lo + i * step for i in range(n)]


def detect_scene_cut_times(path: Path, scene_threshold: float) -> list[float]:
    """Scene cuts via ``select=gt(scene,THRESH),showinfo`` (one decode pass)."""
    ff, _ = require_ffmpeg()
    thr = max(0.01, min(0.99, float(scene_threshold)))
    filt = f"select='gt(scene\\,{thr})',showinfo"
    cmd = [
        ff,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        str(path),
        "-vf",
        filt,
        "-an",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    blob = (proc.stdout or "") + (proc.stderr or "")
    times: list[float] = []
    for m in _PTS_TIME.finditer(blob):
        try:
            times.append(float(m.group(1)))
        except ValueError:
            continue
    return sorted(set(times))


def shot_boundaries(cut_times: list[float], duration: float) -> list[tuple[float, float]]:
    bounds = [0.0] + [t for t in cut_times if 0.0 < t < duration] + [duration]
    bounds = sorted(set(bounds))
    out: list[tuple[float, float]] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a >= 1e-4:
            out.append((a, b))
    return out


def read_rgb_frames(
    path: str | Path,
    *,
    max_frames: int | None = None,
    fps: float | None = None,
    max_width: int | None = None,
) -> tuple[np.ndarray, float]:
    """Decode video to ``uint8`` ``[F, H, W, 3]`` RGB via ffmpeg ``rawvideo`` pipe."""
    pr = probe_media(path)
    if not pr.video:
        raise FFmpegError(f"No video stream in {path}")
    out_fps = fps or pr.video.avg_fps
    out_w, out_h = pr.video.width, pr.video.height
    if max_width and out_w > max_width:
        scale = max_width / out_w
        out_w = int(max_width) - int(max_width) % 2
        out_h = max(2, int(pr.video.height * scale) - int(pr.video.height * scale) % 2)

    vf_parts: list[str] = []
    if max_width:
        vf_parts.append(build_scale_filter(max_width=max_width))
    if fps:
        vf_parts.append(f"fps={fps}")
    vf = ",".join(vf_parts) if vf_parts else None

    ff, _ = require_ffmpeg()
    cmd = [
        ff,
        "-hide_banner",
        "-loglevel",
        os.environ.get("LTX_FFMPEG_LOGLEVEL", "error"),
        *hwaccel_input_args(),
        "-i",
        str(path),
        "-an",
    ]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24"]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-"]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError("ffmpeg rawvideo decode failed", cmd=cmd, stderr=proc.stderr.decode(errors="replace"))
    raw = proc.stdout or b""
    if not raw:
        raise FFmpegError("ffmpeg produced no video data", cmd=cmd)

    frame_bytes = out_w * out_h * 3
    if frame_bytes <= 0 or len(raw) % frame_bytes != 0:
        raise FFmpegError(
            f"Unexpected rawvideo byte count {len(raw)} for {out_w}x{out_h} (frame_bytes={frame_bytes})",
            cmd=cmd,
        )
    n_frames = len(raw) // frame_bytes
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, out_h, out_w, 3).copy()
    return frames, float(out_fps)


def write_rgb_video(
    frames: np.ndarray,
    output_path: str | Path,
    *,
    fps: float = 24.0,
    video_codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    pix_fmt: str = "yuv420p",
    yuv_matrix: Literal["bt601", "bt709", "bt2020"] = "bt709",
    color_range: Literal["tv", "pc"] = "tv",
) -> None:
    """Encode ``[F,H,W,3]`` uint8 RGB to MP4 via rawvideo stdin."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames [F,H,W,3], got {frames.shape}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(frames.shape[1]), int(frames.shape[2])
    ff, _ = require_ffmpeg()
    colorspace = f"colorspace=iall={yuv_matrix}:all={yuv_matrix}:range={color_range}"
    cmd = [
        ff,
        "-hide_banner",
        "-loglevel",
        os.environ.get("LTX_FFMPEG_LOGLEVEL", "error"),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        colorspace,
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        pix_fmt,
        "-an",
        str(out),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError("ffmpeg encode failed", cmd=cmd, stderr=proc.stderr.decode(errors="replace"))


def write_rgb_float32_exr_sequence(
    frames_fhwc: np.ndarray,
    output_pattern: str | Path,
    *,
    fps: float = 24.0,
    loglevel: str | None = None,
) -> None:
    """Write ``[F,H,W,3]`` float32 RGB frames to OpenEXR sequence via ffmpeg.

    Notes:
    - Uses rawvideo stdin to avoid huge temporary uint8 buffers.
    - Uses ``gbrpf32le`` (planar) which is widely supported for EXR.
    """
    if frames_fhwc.ndim != 4 or frames_fhwc.shape[-1] != 3:
        raise ValueError(f"Expected frames [F,H,W,3], got {frames_fhwc.shape}")
    if frames_fhwc.dtype != np.float32:
        frames_fhwc = frames_fhwc.astype(np.float32, copy=False)
    out = Path(output_pattern)
    out.parent.mkdir(parents=True, exist_ok=True)
    f, h, w, _c = frames_fhwc.shape
    ff, _ = require_ffmpeg()
    cmd = [
        ff,
        "-hide_banner",
        "-loglevel",
        (loglevel or os.environ.get("LTX_FFMPEG_LOGLEVEL", "error")),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gbrpf32le",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-f",
        "image2",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for t in range(int(f)):
            frame = frames_fhwc[t]  # [H,W,3] RGB
            # Convert to planar G,B,R as expected by gbrp*
            g = np.ascontiguousarray(frame[..., 1])
            b = np.ascontiguousarray(frame[..., 2])
            r = np.ascontiguousarray(frame[..., 0])
            proc.stdin.write(g.tobytes())
            proc.stdin.write(b.tobytes())
            proc.stdin.write(r.tobytes())
        proc.stdin.close()
        _out, err = proc.communicate()
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    if proc.returncode != 0:
        raise FFmpegError("ffmpeg exr sequence failed", cmd=cmd, stderr=(err or b"").decode(errors="replace"))


def concat_videos(paths: Sequence[Path], output: Path, *, reencode: bool = True) -> None:
    """Concat demuxer; re-encode by default for heterogeneous inputs."""
    if not paths:
        raise ValueError("concat_videos: empty path list")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    list_file = output.with_suffix(".concat.txt")
    lines = [f"file '{p.resolve()}'" for p in paths]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ff, _ = require_ffmpeg()
    cmd = [
        ff,
        "-hide_banner",
        "-loglevel",
        os.environ.get("LTX_FFMPEG_LOGLEVEL", "error"),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
    ]
    if reencode:
        cmd += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(output))
    try:
        _run(cmd)
    finally:
        list_file.unlink(missing_ok=True)


def colorspace_context_from_probe(probe: MediaProbe) -> dict[str, str]:
    """Map ffprobe color fields to yuv_colorspace-friendly meta."""
    v = probe.video
    if v is None:
        return {"yuv_matrix": "bt709", "yuv_range": "limited", "src_pix_fmt": "unknown"}
    matrix = "bt709"
    if v.color_space and "bt601" in v.color_space.lower():
        matrix = "bt601"
    elif v.color_space and "bt2020" in v.color_space.lower():
        matrix = "bt2020"
    elif v.width < 1280 and v.height < 720:
        matrix = "bt601"
    yrange = "full" if v.color_range and "pc" in v.color_range.lower() else "limited"
    return {
        "yuv_matrix": matrix,
        "yuv_range": yrange,
        "src_pix_fmt": v.pix_fmt or "unknown",
        "ffprobe_codec": v.codec_name,
        "color_trc": v.color_transfer or "unknown",
    }
