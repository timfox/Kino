"""Explicit **libdav1d** AV1 decode for LTX video ingest.

PyAV/FFmpeg may pick **libaom-av1** or another slow AV1 decoder depending on the build.
When the stream is AV1 but not already on **libdav1d**, this module demuxes packets and decodes
with a dedicated ``libdav1d`` :class:`~av.codec.CodecContext`.

Enable by default via ``LTX_PREFER_DAV1D=1`` (set ``0``/``false`` to disable). Pass
``prefer_dav1d=False`` to :func:`iter_video_frames` to override per call.

Used by :mod:`ltx_trainer.video_utils` and :mod:`ltx_trainer.hdr_ingest`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av

# Codecs that should be routed through libdav1d when it is not already selected.
_AV1_CODEC_IDS = frozenset(
    {
        "av1",
        "libaom-av1",
        "libsvtav1",
    }
)


@dataclass(frozen=True)
class VideoDecodeInfo:
    """How a file's video track was decoded (for logging / HDR metadata)."""

    backend: str
    codec_name: str


def libdav1d_available() -> bool:
    """Return True when PyAV can construct a ``libdav1d`` decoder."""
    try:
        av.codec.Codec("libdav1d", "r")
        return True
    except av.codec.codec.UnknownCodecError:
        return False


def prefer_dav1d_from_env() -> bool:
    """Read ``LTX_PREFER_DAV1D`` (default on)."""
    raw = os.environ.get("LTX_PREFER_DAV1D", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def resolve_prefer_dav1d(prefer_dav1d: bool | None) -> bool:
    if prefer_dav1d is not None:
        return prefer_dav1d
    return prefer_dav1d_from_env()


def video_stream_codec_name(stream: av.VideoStream) -> str:
    """Normalized codec id for the video stream (e.g. ``libdav1d``, ``h264``)."""
    ctx = stream.codec_context
    if ctx.codec is not None:
        return str(ctx.codec.name).lower()
    if ctx.name:
        return str(ctx.name).lower()
    return "unknown"


def should_decode_av1_via_dav1d(stream: av.VideoStream, *, prefer_dav1d: bool) -> bool:
    """True when we should open an explicit libdav1d context for this stream."""
    if not prefer_dav1d or not libdav1d_available():
        return False
    name = video_stream_codec_name(stream)
    if name == "libdav1d":
        return False
    return name in _AV1_CODEC_IDS or name.startswith("av1")


def open_video_container(video_path: str | Path) -> av.container.InputContainer:
    """Open a video file for reading (thin wrapper for a single import point)."""
    return av.open(str(video_path))


def _copy_stream_params_to_dav1d(src: av.codec.CodecContext, dst: av.codec.CodecContext) -> None:
    """Copy fields needed to decode AV1 bitstream in a fresh libdav1d context."""
    if src.width:
        dst.width = src.width
    if src.height:
        dst.height = src.height
    if src.pix_fmt is not None:
        dst.pix_fmt = src.pix_fmt
    if src.extradata:
        dst.extradata = src.extradata
    if src.thread_count:
        dst.thread_count = src.thread_count


def _create_dav1d_context(stream: av.VideoStream) -> av.codec.CodecContext:
    decoder = av.codec.Codec("libdav1d", "r")
    ctx = av.codec.CodecContext.create(decoder, "r")
    _copy_stream_params_to_dav1d(stream.codec_context, ctx)
    ctx.open()
    return ctx


def _iter_frames_explicit_dav1d(
    container: av.container.InputContainer,
    stream: av.VideoStream,
    *,
    max_frames: int | None,
) -> Iterator[av.VideoFrame]:
    ctx = _create_dav1d_context(stream)
    count = 0
    for packet in container.demux(stream):
        if packet.dts is None and packet.size == 0:
            continue
        for frame in ctx.decode(packet):
            yield frame
            count += 1
            if max_frames is not None and count >= max_frames:
                return
    # Flush decoder
    for frame in ctx.decode(None):
        yield frame
        count += 1
        if max_frames is not None and count >= max_frames:
            return


def _iter_frames_default(
    container: av.container.InputContainer,
    stream: av.VideoStream,
    *,
    max_frames: int | None,
) -> Iterator[av.VideoFrame]:
    count = 0
    for frame in container.decode(stream):
        yield frame
        count += 1
        if max_frames is not None and count >= max_frames:
            return


def iter_video_frames(
    container: av.container.InputContainer,
    *,
    stream_index: int = 0,
    max_frames: int | None = None,
    prefer_dav1d: bool | None = None,
) -> tuple[Iterator[av.VideoFrame], VideoDecodeInfo]:
    """Decode video frames from an opened container.

    Returns ``(iterator, decode_info)``. The iterator must be consumed while ``container`` stays open.
    """
    prefer = resolve_prefer_dav1d(prefer_dav1d)
    stream = container.streams.video[stream_index]
    codec_name = video_stream_codec_name(stream)

    if should_decode_av1_via_dav1d(stream, prefer_dav1d=prefer):
        try:
            return (
                _iter_frames_explicit_dav1d(container, stream, max_frames=max_frames),
                VideoDecodeInfo(backend="libdav1d_explicit", codec_name=codec_name),
            )
        except (av.FFmpegError, OSError, ValueError):
            # Broken extradata / exotic bitstream: fall back to FFmpeg's default decoder.
            pass

    backend = "libdav1d" if codec_name == "libdav1d" else "ffmpeg_default"
    return (
        _iter_frames_default(container, stream, max_frames=max_frames),
        VideoDecodeInfo(backend=backend, codec_name=codec_name),
    )


def decode_info_to_meta(info: VideoDecodeInfo) -> dict[str, Any]:
    """Serialize :class:`VideoDecodeInfo` for HDR / dataset metadata sidecars."""
    return {"video_decode_backend": info.backend, "video_codec": info.codec_name}
