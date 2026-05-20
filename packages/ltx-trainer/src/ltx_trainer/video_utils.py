"""Video I/O utilities for LTX training and dataset prep.

**Decode:** PyAV by default; set ``LTX_VIDEO_BACKEND=ffmpeg`` to use :mod:`ltx_trainer.ffmpeg_io`
(rawvideo pipe, filter graphs, optional hardware decode). AV1 via PyAV can use **libdav1d**
(:mod:`ltx_trainer.dav1d_decode`, ``LTX_PREFER_DAV1D``).

**Probe / cut / scene detect:** use :mod:`ltx_trainer.ffmpeg_io` directly.
"""

import os
from fractions import Fraction
from pathlib import Path
from typing import Literal

import av
import numpy as np
import torch
from ltx_trainer import logger
from ltx_trainer.media_formats import prefer_ffmpeg_video_decode
from torch import Tensor

VideoFormat = Literal["CFHW", "FCHW"]


def _video_backend() -> str:
    return os.environ.get("LTX_VIDEO_BACKEND", "pyav").strip().lower()


def _use_ffmpeg_decode(video_path: str | Path) -> bool:
    if _video_backend() == "ffmpeg":
        return True
    return prefer_ffmpeg_video_decode(video_path)


def _read_video_ffmpeg(video_path: str | Path, max_frames: int | None) -> tuple[Tensor, float]:
    from ltx_trainer.ffmpeg_io import read_rgb_frames

    frames, fps = read_rgb_frames(video_path, max_frames=max_frames)
    video = torch.from_numpy(frames).float().div_(255.0).permute(0, 3, 1, 2)
    return video, fps


def _read_video_pyav(video_path: str | Path, max_frames: int | None) -> tuple[Tensor, float]:
    from ltx_trainer.dav1d_decode import iter_video_frames, open_video_container
    from ltx_trainer.yuv_colorspace import av_frame_to_rgb01, colorspace_context_from_av_stream

    with open_video_container(video_path) as container:
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate or video_stream.base_rate or 24)

        cs_ctx = colorspace_context_from_av_stream(video_stream)
        frames = []
        frame_iter, _decode_info = iter_video_frames(container, max_frames=max_frames)
        for frame in frame_iter:
            rgb01, _ = av_frame_to_rgb01(frame, cs_ctx)
            frames.append(rgb01.permute(1, 2, 0).numpy())

    frames_np = np.stack(frames, axis=0)  # [F, H, W, C]
    video = torch.from_numpy(frames_np).float()
    return video.permute(0, 3, 1, 2), fps  # [F, C, H, W]


def get_video_frame_count(video_path: str | Path) -> int:
    """Get the number of frames in a video file.
    Tries three approaches in order: stream metadata, duration*fps estimate,
    full decode. The estimate may be off by a few frames for VFR videos or
    containers with edit lists — exact for the min_frames filtering use case.
    Args:
        video_path: Path to the video file
    Returns:
        Number of frames in the video
    """
    if _use_ffmpeg_decode(video_path):
        from ltx_trainer.ffmpeg_io import estimate_frame_count

        return estimate_frame_count(video_path)

    try:
        return _get_video_frame_count_pyav(video_path)
    except Exception as exc:
        logger.debug("PyAV frame count failed for %s (%s); using ffprobe", video_path, exc)
        from ltx_trainer.ffmpeg_io import estimate_frame_count

        return estimate_frame_count(video_path)


def _get_video_frame_count_pyav(video_path: str | Path) -> int:
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]

        if video_stream.frames > 0:
            return video_stream.frames

        # Fast estimate from container metadata (avoids full decode).
        # Uses Fraction arithmetic to prevent float precision loss.
        rate = video_stream.average_rate or video_stream.base_rate
        if video_stream.duration and video_stream.time_base and rate:
            duration = Fraction(video_stream.duration) * Fraction(video_stream.time_base)
            return round(duration * Fraction(rate))

        # Last resort: full decode (very slow for 4K)
        from ltx_trainer.dav1d_decode import iter_video_frames

        frame_iter, _ = iter_video_frames(container, stream_index=0)
        return sum(1 for _ in frame_iter)


def read_video(video_path: str | Path, max_frames: int | None = None) -> tuple[Tensor, float]:
    """Load frames from a video file (PyAV by default; ffmpeg for ``.mov`` / on failure).
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to read. If None, reads all frames.
    Returns:
        Video tensor with shape [F, C, H, W] in range [0, 1] and frames per second (fps).
    """
    path = Path(video_path)
    if _use_ffmpeg_decode(path):
        return _read_video_ffmpeg(path, max_frames)
    try:
        return _read_video_pyav(path, max_frames)
    except Exception as exc:
        logger.debug("PyAV decode failed for %s (%s); falling back to ffmpeg", path, exc)
        return _read_video_ffmpeg(path, max_frames)


def save_video(
    video_tensor: torch.Tensor,
    output_path: Path | str,
    fps: float = 24.0,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
    video_format: VideoFormat | None = None,
    *,
    yuv_matrix: str | None = "bt709",
    yuv_range: str | None = "limited",
) -> None:
    """Save a video tensor to a file using PyAV, optionally with audio.
    Args:
        video_tensor: Video tensor of shape [C, F, H, W] or [F, C, H, W] in range [0, 1] or [0, 255]
        output_path: Path to save the video
        fps: Frames per second for the output video
        audio: Optional audio tensor of shape [C, samples] or [samples, C] in range [-1, 1]
        audio_sample_rate: Sample rate for the audio (required if audio is provided)
        video_format: Explicit layout of ``video_tensor``, either ``"CFHW"`` or ``"FCHW"``.
            When ``None`` (default), the layout is auto-detected using a heuristic that only
            works when ``shape[1] > 3`` — the ambiguous ``[C=3, F=3, H, W]`` / ``[F=3, C=3, H, W]``
            case requires passing this argument explicitly.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to [F, H, W, C] uint8 numpy array
    video_np = _prepare_video_array(video_tensor, video_format=video_format)
    _, height, width, _ = video_np.shape

    with av.open(str(output_path), mode="w") as container:
        # Setup video stream
        video_stream = container.add_stream("libx264", rate=int(fps))
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": "18"}

        # Setup audio stream if needed
        if audio is not None:
            if audio_sample_rate is None:
                raise ValueError("audio_sample_rate must be provided when audio is given")
            audio_stream = container.add_stream("aac", rate=audio_sample_rate)
            audio_stream.layout = "stereo"
            audio_stream.time_base = Fraction(1, audio_sample_rate)

        from ltx_trainer.yuv_colorspace import YuvMatrix, YuvRange, rgb01_fhw_to_av_yuv420p_frame

        matrix = YuvMatrix(yuv_matrix) if yuv_matrix else YuvMatrix.BT709
        yrange = YuvRange(yuv_range) if yuv_range else YuvRange.LIMITED

        # Write video frames (RGB uint8 → explicit yuv420p, then libx264 — ffmpeg-style path)
        for frame_array in video_np:
            rgb01 = torch.from_numpy(frame_array).float().permute(2, 0, 1).div_(255.0)
            frame = rgb01_fhw_to_av_yuv420p_frame(rgb01, matrix=matrix, yuv_range=yrange)
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        # Write audio if provided
        if audio is not None:
            _write_audio(container, audio_stream, audio, audio_sample_rate)


def save_video_ffmpeg(
    video_tensor: torch.Tensor,
    output_path: Path | str,
    fps: float = 24.0,
    video_format: VideoFormat | None = None,
    *,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """Encode via :mod:`ltx_trainer.ffmpeg_io` (colorspace filter + libx264). No audio."""
    from ltx_trainer.ffmpeg_io import write_rgb_video

    video_np = _prepare_video_array(video_tensor, video_format=video_format)
    write_rgb_video(video_np, output_path, fps=fps, crf=crf, preset=preset)


def _prepare_video_array(
    video_tensor: torch.Tensor,
    video_format: VideoFormat | None = None,
) -> np.ndarray:
    """Convert video tensor to [F, H, W, C] uint8 numpy array.
    If ``video_format`` is provided, it is trusted. Otherwise, the layout is auto-detected
    using a heuristic that only fires when ``shape[0] == 3 and shape[1] > 3`` (CFHW). The
    ambiguous ``[C=3, F=3, H, W]`` / ``[F=3, C=3, H, W]`` case cannot be disambiguated and
    defaults to the FCHW interpretation — callers must pass ``video_format`` explicitly for
    3-frame CFHW tensors.
    """
    if video_format == "CFHW":
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, F, H, W] -> [F, C, H, W]
    elif video_format is None and video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:
        video_tensor = video_tensor.permute(1, 0, 2, 3)

    # Normalize to [0, 255] uint8
    if video_tensor.max() <= 1.0:
        video_tensor = video_tensor * 255

    # [F, C, H, W] -> [F, H, W, C]
    return video_tensor.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()


def _write_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    audio: torch.Tensor,
    sample_rate: int,
) -> None:
    """Write audio tensor to container as stereo AAC."""
    audio = audio.cpu().float()

    # Normalize to [samples, 2] stereo format
    if audio.ndim == 1:
        audio = audio.unsqueeze(1).repeat(1, 2)  # Mono -> stereo
    elif audio.shape[0] == 2 and audio.shape[1] != 2:
        audio = audio.T  # [2, samples] -> [samples, 2]
    if audio.shape[1] == 1:
        audio = audio.repeat(1, 2)  # Mono -> stereo

    # Convert to int16 interleaved: [samples, 2] -> [1, samples*2]
    audio_int16 = (audio.clamp(-1, 1) * 32767).to(torch.int16)
    audio_interleaved = audio_int16.contiguous().view(1, -1).numpy()

    # Create audio frame
    frame = av.AudioFrame.from_ndarray(audio_interleaved, format="s16", layout="stereo")
    frame.sample_rate = sample_rate

    # Resample to encoder format and write
    resampler = av.audio.resampler.AudioResampler(
        format=audio_stream.codec_context.format,
        layout=audio_stream.codec_context.layout,
        rate=sample_rate,
    )

    pts = 0
    for resampled_frame in resampler.resample(frame):
        resampled_frame.pts = pts
        pts += resampled_frame.samples
        for packet in audio_stream.encode(resampled_frame):
            container.mux(packet)

    for packet in audio_stream.encode():
        container.mux(packet)
