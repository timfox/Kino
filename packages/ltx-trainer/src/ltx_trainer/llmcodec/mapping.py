"""Weight → YUV420 mapping and RTN quantize/dequantize (arXiv:2606.05861 §III-C)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.llmcodec.config import LlmCodecConfig, YuvLayout


def _as_f64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def rt_quantize(W: np.ndarray, *, bits: int = 8) -> tuple[np.ndarray, float]:
    """Round-to-nearest INT8 mapping with per-tensor scale s."""
    w = _as_f64(W)
    max_abs = float(np.max(np.abs(w))) or 1.0
    qmax = (1 << (bits - 1)) - 1
    scale = max_abs / qmax
    w_q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int16)
    return w_q, scale


def rt_dequantize(Wq: np.ndarray, scale: float) -> np.ndarray:
    return _as_f64(Wq) * float(scale)


def weight_matrix_to_y_plane(
    Wq: np.ndarray,
    *,
    height: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    """Reshape quantized weights into a 2D Y plane (luminance-only storage)."""
    flat = Wq.reshape(-1)
    n = flat.size
    if height is None or width is None:
        side = int(np.ceil(np.sqrt(n)))
        height = width = side
    pad = height * width - n
    if pad > 0:
        flat = np.pad(flat, (0, pad), mode="constant")
    y = flat[: height * width].reshape(height, width)
    # map signed int to 0..255 for codec input
    y_u8 = np.clip(y.astype(np.int32) + 128, 0, 255).astype(np.uint8)
    return y_u8


def y_plane_to_weights(y_u8: np.ndarray, *, original_shape: tuple[int, ...], scale: float) -> np.ndarray:
    signed = y_u8.astype(np.int16) - 128
    flat = signed.reshape(-1)[: int(np.prod(original_shape))]
    w_q = flat.reshape(original_shape)
    return rt_dequantize(w_q, scale)


def pack_yuv420(
    y_plane: np.ndarray,
    *,
    cfg: LlmCodecConfig | None = None,
    u_val: int | None = None,
    v_val: int | None = None,
) -> dict[str, np.ndarray]:
    """Y full-res; U/V constant (paper: negligible bitrate)."""
    c = cfg or LlmCodecConfig()
    h, w = y_plane.shape
    fill = c.chroma_fill if u_val is None else u_val
    vfill = c.chroma_fill if v_val is None else v_val
    if c.yuv_layout == YuvLayout.YUV444:
        return {
            "Y": y_plane,
            "U": np.full((h, w), fill, dtype=np.uint8),
            "V": np.full((h, w), fill, dtype=np.uint8),
        }
    uh, uw = (h + 1) // 2, (w + 1) // 2
    return {
        "Y": y_plane,
        "U": np.full((uh, uw), fill, dtype=np.uint8),
        "V": np.full((uh, uw), fill, dtype=np.uint8),
    }


def weight_to_yuv420(
    W: np.ndarray,
    *,
    cfg: LlmCodecConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Full pipeline: affine-ready float W → INT8 → YUV420 planes."""
    c = cfg or LlmCodecConfig()
    w_q, scale = rt_quantize(W, bits=c.int_bits)
    y = weight_matrix_to_y_plane(w_q)
    yuv = pack_yuv420(y, cfg=c)
    meta = {"scale": scale, "original_shape": tuple(W.shape), "y_shape": y.shape}
    return yuv, meta
