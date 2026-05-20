"""FFmpeg-style YUV ↔ RGB conversion for LTX video ingest and export.

Decoders (H.264, HEVC, etc.) usually surface **yuv420p** planes. Converting with the wrong
matrix or range (limited TV vs full PC) shifts colors before the VAE sees them. This module
mirrors the common ``libswscale`` / ``colorspace`` filter behavior:

- Matrices: BT.601 (SD), BT.709 (HD), BT.2020 (UHD / HDR containers)
- Range: MPEG/TV limited (16–235 luma) vs JPEG/full (0–255)
- Chroma: 4:2:0 planar upsampled with bilinear (default swscale-style)

PyAV ``VideoFrame`` objects in ``yuv420p`` are converted to RGB float01 ``[3, H, W]`` for
:func:`ltx_trainer.hdr_ingest.read_video_hdr_float32` and :func:`ltx_trainer.video_utils.read_video`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class YuvMatrix(str, Enum):
    BT601 = "bt601"
    BT709 = "bt709"
    BT2020 = "bt2020"


class YuvRange(str, Enum):
    """Chroma/luma coding range (ffmpeg ``color_range``)."""

    LIMITED = "limited"  # MPEG / TV — Y in [16, 235], Cb/Cr in [16, 240]
    FULL = "full"  # JPEG / PC — full 8-bit swing


@dataclass(frozen=True)
class ColorspaceContext:
    matrix: YuvMatrix
    yuv_range: YuvRange
    src_pix_fmt: str

    def to_meta(self) -> dict[str, str]:
        return {
            "yuv_matrix": self.matrix.value,
            "yuv_range": self.yuv_range.value,
            "src_pix_fmt": self.src_pix_fmt,
        }


# ITU-R RGB ← YCbCr: R = Y + r_cr*Cr, G = Y + g_cb*Cb + g_cr*Cr, B = Y + b_cb*Cb (Cr/Cb ∈ [-0.5, 0.5]).
_MATRIX_COEFF: dict[YuvMatrix, tuple[float, float, float, float]] = {
    YuvMatrix.BT601: (1.402, -0.344136, -0.714136, 1.772),
    YuvMatrix.BT709: (1.5748, -0.187324, -0.468124, 1.8556),
    YuvMatrix.BT2020: (1.4746, -0.164553, -0.571353, 1.8814),
}


def _coeff(matrix: YuvMatrix) -> tuple[float, float, float, float]:
    return _MATRIX_COEFF[matrix]


def _normalize_y(y: Tensor, yuv_range: YuvRange) -> Tensor:
    if yuv_range == YuvRange.FULL:
        return y.clamp(0.0, 1.0)
    # Limited: 16/255 .. 235/255 → [0, 1]
    return ((y - 16.0 / 255.0) / (219.0 / 255.0)).clamp(0.0, 1.0)


def _normalize_c(c: Tensor, yuv_range: YuvRange) -> Tensor:
    """Cb or Cr plane → centered chroma in [-0.5, 0.5]."""
    if yuv_range == YuvRange.FULL:
        return (c - 0.5).clamp(-0.5, 0.5)
    return ((c - 128.0 / 255.0) / (224.0 / 255.0)).clamp(-0.5, 0.5)


def _upsample_chroma(plane: Tensor, height: int, width: int) -> Tensor:
    """Bilinear 2× upsample of a ``[H/2, W/2]`` chroma plane to ``[H, W]``."""
    x = plane.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return up.squeeze(0).squeeze(0)


def yuv420p_planes_to_rgb01(
    y_plane: Tensor,
    u_plane: Tensor,
    v_plane: Tensor,
    *,
    matrix: YuvMatrix = YuvMatrix.BT709,
    yuv_range: YuvRange = YuvRange.LIMITED,
) -> Tensor:
    """Convert 4:2:0 planar Y, U, V to RGB float01 ``[3, H, W]``."""
    h, w = y_plane.shape[-2], y_plane.shape[-1]
    y_n = _normalize_y(y_plane.float(), yuv_range)
    cb = _normalize_c(_upsample_chroma(u_plane.float(), h, w), yuv_range)
    cr = _normalize_c(_upsample_chroma(v_plane.float(), h, w), yuv_range)
    r_cr, g_cb, g_cr, b_cb = _coeff(matrix)
    r = (y_n + r_cr * cr).clamp(0.0, 1.0)
    g = (y_n + g_cb * cb + g_cr * cr).clamp(0.0, 1.0)
    b = (y_n + b_cb * cb).clamp(0.0, 1.0)
    return torch.stack([r, g, b], dim=0)


def rgb01_chw_to_yuv420p_planes(
    rgb: Tensor,
    *,
    matrix: YuvMatrix = YuvMatrix.BT709,
    yuv_range: YuvRange = YuvRange.LIMITED,
) -> tuple[Tensor, Tensor, Tensor]:
    """RGB float01 ``[3, H, W]`` → downsampled Y, U, V planes (4:2:0)."""
    r, g, b = rgb[0].float(), rgb[1].float(), rgb[2].float()
    r_cr, g_cb, g_cr, b_cb = _coeff(matrix)
    # Inverse of RGB = Y + k*C (approximate RGB→YCbCr for encoding).
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b) if matrix == YuvMatrix.BT709 else (
        (0.299 * r + 0.587 * g + 0.114 * b) if matrix == YuvMatrix.BT601
        else (0.2627 * r + 0.6780 * g + 0.0593 * b)
    )
    cb = (b - y) / b_cb
    cr = (r - y) / r_cr
    h, w = y.shape
    cb_ds = F.avg_pool2d(cb.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()
    cr_ds = F.avg_pool2d(cr.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()

    if yuv_range == YuvRange.FULL:
        y_out = y.clamp(0.0, 1.0)
        u_out = (cb_ds + 0.5).clamp(0.0, 1.0)
        v_out = (cr_ds + 0.5).clamp(0.0, 1.0)
    else:
        y_out = (y * (219.0 / 255.0) + 16.0 / 255.0).clamp(0.0, 1.0)
        u_out = (cb_ds * (224.0 / 255.0) + 128.0 / 255.0).clamp(0.0, 1.0)
        v_out = (cr_ds * (224.0 / 255.0) + 128.0 / 255.0).clamp(0.0, 1.0)
    return y_out, u_out, v_out


def _enum_name(val: Any) -> str:
    if val is None:
        return "unspecified"
    name = getattr(val, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(int(val) if val is not None else -1)


def infer_yuv_matrix(
    *,
    color_primaries: Any = None,
    colorspace: Any = None,
    width: int = 1920,
    height: int = 1080,
) -> YuvMatrix:
    """Pick matrix like ffmpeg when metadata is missing (HD → BT.709, SD → BT.601)."""
    for tag in (_enum_name(color_primaries), _enum_name(colorspace)):
        if "2020" in tag or "bt2020" in tag:
            return YuvMatrix.BT2020
        if "709" in tag or "bt709" in tag:
            return YuvMatrix.BT709
        if "470" in tag or "601" in tag or "smpte170" in tag or "bt601" in tag:
            return YuvMatrix.BT601
    if min(width, height) >= 720 or max(width, height) >= 1280:
        return YuvMatrix.BT709
    return YuvMatrix.BT601


def infer_yuv_range(color_range: Any = None) -> YuvRange:
    tag = _enum_name(color_range)
    if "jpeg" in tag or "pc" in tag or "full" in tag:
        return YuvRange.FULL
    # unspecified / mpeg / tv → limited for broadcast-style yuv420p
    return YuvRange.LIMITED


def colorspace_context_from_av_stream(stream: Any) -> ColorspaceContext:
    ctx = stream.codec_context
    w = int(ctx.width or 0)
    h = int(ctx.height or 0)
    matrix = infer_yuv_matrix(
        color_primaries=getattr(ctx, "color_primaries", None),
        colorspace=getattr(ctx, "colorspace", None),
        width=w,
        height=h,
    )
    yuv_range = infer_yuv_range(getattr(ctx, "color_range", None))
    pix = str(getattr(ctx, "pix_fmt", None) or "unknown")
    return ColorspaceContext(matrix=matrix, yuv_range=yuv_range, src_pix_fmt=pix)


def _plane_to_tensor(plane: Any, *, width: int, height: int) -> Tensor:
    """Read a PyAV ``VideoPlane`` buffer into ``[height, width]`` float01."""
    line = int(plane.line_size)
    buf = np.frombuffer(plane, dtype=np.uint8, count=line * height)
    arr = buf.reshape(height, line)[:, :width].copy()
    return torch.from_numpy(arr).float().div_(255.0)


def _planes_from_av_frame(frame: av.VideoFrame) -> tuple[Tensor, Tensor, Tensor]:
    fmt = (frame.format.name or "").lower()
    if fmt != "yuv420p":
        frame = frame.reformat(format="yuv420p")
    h, w = frame.height, frame.width
    y = _plane_to_tensor(frame.planes[0], width=w, height=h)
    uh, uw = h // 2, w // 2
    u = _plane_to_tensor(frame.planes[1], width=uw, height=uh)
    v = _plane_to_tensor(frame.planes[2], width=uw, height=uh)
    return y, u, v


def av_frame_to_rgb01(
    frame: av.VideoFrame,
    ctx: ColorspaceContext | None = None,
) -> tuple[Tensor, ColorspaceContext]:
    """Decode one PyAV frame to RGB float01 ``[3, H, W]`` with explicit YUV math when possible."""
    fmt = (frame.format.name or "").lower()
    if fmt == "yuv420p" or fmt == "yuvj420p":
        y, u, v = _planes_from_av_frame(frame)
        if ctx is None:
            # yuvj420p signals full-range JPEG
            yuv_range = YuvRange.FULL if "j" in fmt else YuvRange.LIMITED
            ctx = ColorspaceContext(matrix=YuvMatrix.BT709, yuv_range=yuv_range, src_pix_fmt=fmt)
        rgb = yuv420p_planes_to_rgb01(y, u, v, matrix=ctx.matrix, yuv_range=ctx.yuv_range)
        return rgb, ctx
    # Fallback: libav RGB pack (may not match HDR matrix intent).
    try:
        conv = frame.reformat(format="rgb48le")
        arr = conv.to_ndarray()
        t = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1).div_(65535.0)
    except Exception:
        arr = frame.to_ndarray(format="rgb24")
        t = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1).div_(255.0)
    fb_ctx = ctx or ColorspaceContext(matrix=YuvMatrix.BT709, yuv_range=YuvRange.FULL, src_pix_fmt=fmt or "rgb")
    return t, fb_ctx


def rgb01_fhw_to_av_yuv420p_frame(
    rgb_fhw: Tensor,
    *,
    matrix: YuvMatrix = YuvMatrix.BT709,
    yuv_range: YuvRange = YuvRange.LIMITED,
) -> av.VideoFrame:
    """Build a ``yuv420p`` PyAV frame from ``[F, 3, H, W]`` or ``[3, H, W]`` RGB float01 (first frame if 4D)."""
    if rgb_fhw.ndim == 4:
        rgb = rgb_fhw[0]
    else:
        rgb = rgb_fhw
    y, u, v = rgb01_chw_to_yuv420p_planes(rgb, matrix=matrix, yuv_range=yuv_range)
    h, w = y.shape
    frame = av.VideoFrame(width=w, height=h, format="yuv420p")
    y_np = (y.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    u_np = (u.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    v_np = (v.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    frame.planes[0].update(y_np)
    frame.planes[1].update(u_np)
    frame.planes[2].update(v_np)
    return frame
