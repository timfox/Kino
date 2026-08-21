"""Video-codec compression proxy for YUV weight frames (arXiv:2606.05861)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.llmcodec.config import CodecType, CodingProfile, LlmCodecConfig


@dataclass
class CodecResult:
    y_reconstructed: np.ndarray
    psnr_y: float
    psnr_u: float
    psnr_v: float
    bytes_estimate: int
    qp: int
    codec: str
    profile: str


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)
    mse = float(np.mean((o - r) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(255.0**2 / mse))


def _profile_penalty(profile: CodingProfile) -> float:
    """All-Intra best for static weight tensors (paper Fig. 4b)."""
    if profile == CodingProfile.ALL_INTRA:
        return 0.0
    if profile == CodingProfile.LOW_DELAY:
        return 2.5
    return 3.5


def _codec_efficiency(codec: CodecType) -> float:
    """Relative RD efficiency vs VVC at same QP (paper Fig. 4c)."""
    return {
        CodecType.VVC: 1.0,
        CodecType.HEVC: 0.92,
        CodecType.WEBP: 0.78,
        CodecType.JPEG: 0.72,
    }[codec]


def compress_yuv420_stub(
    yuv: dict[str, np.ndarray],
    *,
    qp: int = 12,
    codec: CodecType | None = None,
    profile: CodingProfile | None = None,
    cfg: LlmCodecConfig | None = None,
) -> CodecResult:
    """Simulate VVenC All-Intra: QP-controlled quant noise + byte estimate."""
    c = cfg or LlmCodecConfig()
    codec = codec or c.default_codec
    profile = profile or c.default_profile
    y = yuv["Y"]
    u = yuv["U"]
    v = yuv["V"]

    eff = _codec_efficiency(codec)
    noise_sigma = max(0.5, (qp + _profile_penalty(profile)) / eff * 0.35)
    rng = np.random.default_rng(qp * 17 + hash(codec.value) % 997)
    y_hat = np.clip(y.astype(np.float64) + rng.normal(0, noise_sigma, size=y.shape), 0, 255).astype(np.uint8)
    u_hat = u.copy()
    v_hat = v.copy()

    y_bits = float(np.std(y.astype(np.float64))) + 1.0
    # Higher QP → smaller bitstream (more compression).
    bytes_est = int(max(64, y.size * max(2.0, (22 - qp)) / (eff * 6.0)))

    return CodecResult(
        y_reconstructed=y_hat,
        psnr_y=round(psnr(y, y_hat), 3),
        psnr_u=round(psnr(u, u_hat), 3),
        psnr_v=round(psnr(v, v_hat), 3),
        bytes_estimate=bytes_est,
        qp=int(qp),
        codec=codec.value,
        profile=profile.value,
    )


def effective_bitwidth(n_params: int, total_bytes: int) -> float:
    return 8.0 * total_bytes / max(n_params, 1)


def rd_curve_stub(
    *,
    qp_values: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
    codec: CodecType = CodecType.VVC,
    profile: CodingProfile = CodingProfile.ALL_INTRA,
    seed: int = 0,
) -> list[dict[str, float | int]]:
    """Synthetic R-D points for a k-projection layer (Fig. 4)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, size=(256, 256))
    from ltx_trainer.llmcodec.mapping import weight_to_yuv420

    yuv, _ = weight_to_yuv420(w)
    points: list[dict[str, float | int]] = []
    for qp in qp_values:
        res = compress_yuv420_stub(yuv, qp=qp, codec=codec, profile=profile)
        bw = effective_bitwidth(w.size, res.bytes_estimate)
        points.append({"qp": qp, "psnr_y": res.psnr_y, "avg_bitwidth": round(bw, 3)})
    return points
