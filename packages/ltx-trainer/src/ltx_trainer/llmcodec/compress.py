"""End-to-end LLMCodec layer compress/decompress stub."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.llmcodec.affine import apply_affine, learn_affine_stub
from ltx_trainer.llmcodec.codec import CodecResult, compress_yuv420_stub, effective_bitwidth, psnr
from ltx_trainer.llmcodec.config import CodecType, CodingProfile, LlmCodecConfig
from ltx_trainer.llmcodec.mapping import rt_dequantize, rt_quantize, weight_to_yuv420, y_plane_to_weights


def compress_weight_matrix(
    W: np.ndarray,
    *,
    qp: int = 12,
    codec: CodecType | None = None,
    profile: CodingProfile | None = None,
    use_affine: bool = True,
    cfg: LlmCodecConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Affine → RTN → YUV420 → codec → reconstruct weights."""
    c = cfg or LlmCodecConfig()
    w = np.asarray(W, dtype=np.float64)
    t = None
    w_prep = w
    affine_err = 0.0
    if use_affine:
        t, affine_err = learn_affine_stub(w, seed=seed)
        w_prep = apply_affine(w, t)

    w_q, scale = rt_quantize(w_prep, bits=c.int_bits)
    yuv, meta = weight_to_yuv420(w_prep, cfg=c)
    codec_res: CodecResult = compress_yuv420_stub(yuv, qp=qp, codec=codec, profile=profile, cfg=c)
    w_hat = y_plane_to_weights(codec_res.y_reconstructed, original_shape=meta["original_shape"], scale=scale)

    weight_psnr = psnr(
        np.clip(w_q.astype(np.float64) + 128, 0, 255),
        codec_res.y_reconstructed.astype(np.float64),
    )
    n_params = w.size
    avg_bw = effective_bitwidth(n_params, codec_res.bytes_estimate)

    return {
        "n_params": n_params,
        "qp": qp,
        "codec": codec_res.codec,
        "profile": codec_res.profile,
        "psnr_y": codec_res.psnr_y,
        "weight_psnr": round(weight_psnr, 3),
        "bytes_estimate": codec_res.bytes_estimate,
        "avg_bitwidth": round(avg_bw, 3),
        "affine_error": round(affine_err, 5),
        "use_affine": use_affine,
        "recon_mse": float(np.mean((w_prep - w_hat) ** 2)),
    }


def decompress_weight_matrix(
    codec_result: CodecResult,
    meta: dict[str, Any],
) -> np.ndarray:
    return y_plane_to_weights(codec_result.y_reconstructed, original_shape=meta["original_shape"], scale=meta["scale"])
