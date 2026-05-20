"""HDR-oriented video/still decode: float32 tensors and HDR research helpers (LatentHDR, X2HDR).

Decodes frames to **scene-linear** RGB float32 (relative intensity; PQ/HLG paths follow
ITU-R BT.2100-style formulas). Intended for auxiliary ``hdr_latent`` storage alongside
VAE latents, not as a replacement for the VAE's trained input domain.

**Alignment with LatentHDR** (arXiv:2605.11115, non-exclusive arXiv license):

- **Scene anchor vs exposure**: ``hdr_latent`` is treated as a shared scene-radiance proxy
  (EV=0 anchor up to per-clip normalization). Exposure variation is modeled as a
  *deterministic* radiometric scaling in linear space, not as repeated stochastic diffusion.
- **Synthetic LDR brackets** (Appendix A): ``synthetic_gamma_ldr_stack_from_linear_hdr`` applies
  ``x_e = clip(x_hdr * 2^e, 0, 1)``, then ``y_e = x_e ** (1/gamma)`` (default ``gamma=2.2``),
  matching the paper's training bracket construction (before 8-bit quantization).
- **HDR merge** (Sec. 3.5, Eq. 9): ``merge_log_domain_radiance`` implements log-domain fusion
  with per-exposure validity masks and triangular weights (simplified from the paper).
- **VAE latents as posterior means**: LTX's ``VideoEncoder.forward`` returns normalized
  **means** only (log-var is stripped after ``torch.chunk``), matching LatentHDR's use of
  ``μ(x)`` rather than sampled ``z`` for supervision when variance is negligible.

**Alignment with X2HDR** (arXiv:2602.04814, arXiv perpetual non-exclusive license):

- **Perceptually uniform VAE input**: LDR-pretrained VAEs match statistics better when HDR is
  encoded in **PU21** (Mantiuk and Azimi, 2021) than in naive linear RGB. Optional
  ``hdr_vae_encoding="pu21"`` maps scene-linear radiance (after per-clip peak rescale to
  ``L_peak`` cd/m²) through the paper's log-quadratic PU21 forward, then the usual
  ``[0,1] → [-1,1]`` normalization for the frozen VAE (Sec. 3--4).
- **Inverse**: ``pu21_inverse_to_linear_abs`` recovers absolute linear luminance from PU21
  codes for offline inspection or future decode pipelines (Eq. inverse in Sec. 3.1).
- **FP32**: X2HDR notes BF16 can band in smooth dark gradients (Appendix D.10); this repo
  already defaults saved VAE latents to float32 where configured.

**Alignment with LumiVid** (arXiv:2604.11788, CC BY 4.0):

- **Logarithmic / camera-oriented VAE input**: LumiVid argues a fixed **ARRI LogC3** curve aligns
  scene-linear HDR with the **pixel and latent** statistics of SDR-pretrained video models better than
  several display-centric transforms, enabling **frozen VAE** + lightweight **LoRA** adaptation.
- **This repo**: optional ``hdr_vae_encoding="logc3"`` feeds ``linear_scene_to_logc3_display`` into the
  same ``[0,1] → [-1,1]`` path as other encodings. Inverse for decode / EXR export reuses
  ``ltx_core.hdr.LogC3.decompress`` (same implementation as ``tools/logc3_numpy.py``).
- **Training signal** (camera-mimicking degradations on SDR references) is a **trainer** concern; it is
  not applied in latent preprocessing here—only noted in ``lumivid_meta_block`` for provenance.

**Alignment with LF-Diff** (arXiv:2404.00849, arXiv perpetual non-exclusive license):

- **Compact target for diffusion**: LF-Diff applies the DM to a **small low-frequency prior** (LPR), not
  to a full HDR image from pure noise, and fuses it with a **regression** HDR head—much cheaper than
  pixel-space DiffHDR-style sampling. That split is a **trainer / architecture** choice; this repo only
  documents it in ``lf_diff_meta_block``.
- **Tonemap used in the paper** for LPENet inputs and reconstruction loss: ``T(x)=log(1+μ x)/log(1+μ)``
  with ``μ=5000`` (Sec. 4.1). Optional ``hdr_vae_encoding="lf_log1p"`` uses the same ``T`` on
  non-negative scene-linear frames as a bounded ``[0,1]`` proxy before the usual VAE normalize.
- **Joint training** (DM + DHRNet) vs split training is noted in metadata; not enforced here.
- **Reconstruction loss (Eq. 11)**: tonemapped L1 ``||T(H)-T(H_pred)||_1`` plus VGG perceptual term on
  ``T(H)``; use :func:`lf_diff_tonemap_l1` for the first term when both tensors are non-negative linear HDR.
- **LPENet stack (Eq. 6)**: :func:`lf_diff_concat_linear_tonemap` builds ``Concat(H, T(H))`` on channels; apply
  ``torch.nn.PixelUnshuffle`` downstream when training an LPENet-style encoder.
- **FRM split (Eq. 7–8)**: :func:`lf_diff_feature_split_low_high` separates avg-pooled low frequency from a
  high-frequency residual (default ``k=4`` like PIM; pass ``k=2`` for FRM-scale pooling in the paper).
- **Stage-2 prior loss (Eq. 15)**: :func:`lf_diff_lpr_l1` is the ``||z_hat - z||_1`` mean term alongside DM noise MSE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from ltx_core.hdr import LogC3
from torch import Tensor


# X2HDR (arXiv:2602.04814) PU21 log-quadratic encoding constants (Sec. 3.1, Ke et al. 2023 fit).
_PU21_A = 0.001908
_PU21_B = 0.0078
_PU21_LOG2_LMIN = math.log2(0.005)
X2HDR_DEFAULT_L_PEAK_CD_M2 = 4000.0


def pu21_forward_linear_abs(L: Tensor) -> Tensor:
    """Map absolute linear luminance ``L`` (cd/m² scale, per channel) to PU21 code ``V`` in ``[0, 1]``.

    Uses ``V = a (log2 L - Lmin)^2 + b (log2 L - Lmin)`` with ``L ∈ [0.005, 10000]`` (X2HDR Sec. 3.1).
    """
    Lc = L.clamp(0.005, 10000.0)
    u = torch.log2(Lc) - _PU21_LOG2_LMIN
    return (_PU21_A * u * u + _PU21_B * u).clamp(0.0, 1.0)


def pu21_inverse_to_linear_abs(V: Tensor) -> Tensor:
    """Inverse PU21: ``V ∈ [0,1]`` → absolute linear ``L`` in ``[0.005, 10000]``."""
    v = V.clamp(0.0, 1.0)
    disc = _PU21_B * _PU21_B + 4.0 * _PU21_A * v
    u = (-_PU21_B + torch.sqrt(disc.clamp(min=0.0))) / (2.0 * _PU21_A)
    log_l = u + _PU21_LOG2_LMIN
    return torch.pow(2.0, log_l).clamp(0.005, 10000.0)


def linear_scene_to_pu21_display(
    scene_linear_fchw: Tensor,
    *,
    l_peak: float = X2HDR_DEFAULT_L_PEAK_CD_M2,
) -> Tensor:
    """X2HDR-style PU21 **display** tensor for a frozen LDR VAE (same shape as ``scene_linear_fchw``).

    Globally rescales positive scene-linear RGB so the tensor maximum maps to ``l_peak`` (paper:
    4,000 cd/m²), then applies ``f_PU21`` channel-wise. Output is in ``[0, 1]`` for each channel,
    suitable for the same ``clamp + Normalize(0.5)`` stack as Reinhard-tonemapped SDR proxies.
    """
    x = scene_linear_fchw.clamp(min=0.0)
    peak = x.amax().clamp(min=1e-10)
    L_abs = x * (l_peak / peak)
    return pu21_forward_linear_abs(L_abs)


def x2hdr_meta_block(*, l_peak: float = X2HDR_DEFAULT_L_PEAK_CD_M2, vae_encoding: str = "pu21") -> dict[str, Any]:
    """Compact X2HDR provenance for ``hdr_meta`` JSON."""
    return {
        "x2hdr": {
            "arxiv_id": "2602.04814",
            "vae_input_encoding": vae_encoding,
            "L_peak_cd_m2": l_peak,
            "pu21_params": {"a": _PU21_A, "b": _PU21_B, "log2_Lmin": _PU21_LOG2_LMIN},
        }
    }


_LOGC3 = LogC3()


def linear_scene_to_logc3_display(scene_linear_fchw: Tensor) -> Tensor:
    """LumiVid-style **LogC3** codes in ``[0, 1]`` for frozen SDR VAE input (arXiv:2604.11788, Sec. 3.1).

    Maps non-negative scene-linear RGB with ARRI LogC3 (EI 800 constants in ``ltx_core.hdr.LogC3``).
    Unlike X2HDR PU21 here, **no per-clip peak rescale** is applied: LogC3 already spans many orders
    of magnitude in linear light; rescaling would distort relative radiance vs the standard curve.
    Very bright linear values can **saturate** to LogC3 code 1.0; inverse decompress then cannot recover
    the original peak (same as ``ltx_core.hdr.LogC3`` semantics).
    """
    return _LOGC3.compress(scene_linear_fchw.clamp(min=0.0))


def lumivid_meta_block(*, vae_encoding: str = "logc3") -> dict[str, Any]:
    """LumiVid / latent-alignment provenance for ``hdr_meta`` JSON."""
    return {
        "lumivid": {
            "arxiv_id": "2604.11788",
            "vae_input_encoding": vae_encoding,
            "logc3": "ltx_core.hdr.LogC3 (EI 800)",
            "training_note": "Paper Sec. 3.2: camera-mimicking degradations on SDR reference are separate from this preprocess.",
        }
    }


# LF-Diff (arXiv:2404.00849) tonemap for LPENet / loss (Sec. 4.1), μ controls highlight compression.
LF_DIFF_TONEMAP_MU_DEFAULT = 5000.0


def lf_diff_tonemap_display(x: Tensor, *, mu: float = LF_DIFF_TONEMAP_MU_DEFAULT) -> Tensor:
    """LF-Diff operator ``T(x) = log(1 + μ x) / log(1 + μ)`` for ``x ≥ 0``, output in ``[0, 1]``."""
    xc = x.clamp(min=0.0, max=1e6)
    return (torch.log1p(mu * xc) / math.log1p(mu)).clamp(0.0, 1.0)


def lf_diff_tonemap_inverse_display(y: Tensor, *, mu: float = LF_DIFF_TONEMAP_MU_DEFAULT) -> Tensor:
    """Inverse of :func:`lf_diff_tonemap_display`: ``[0,1]`` codes back to linear ``x ≥ 0``.

    Note: ``y → 1`` corresponds to large linear ``x`` at fixed ``μ``; saturated codes cannot recover
    peaks above ``x ≈ 1`` in normalized units (``T(1)=1`` for the paper's ``μ=5000``).
    """
    yc = y.clamp(0.0, 1.0)
    return torch.expm1(yc * math.log1p(mu)) / mu


def linear_scene_to_lf_diff_tonemap_display(
    scene_linear_fchw: Tensor,
    *,
    mu: float = LF_DIFF_TONEMAP_MU_DEFAULT,
) -> Tensor:
    """Apply LF-Diff ``T`` channel-wise to scene-linear frames for frozen VAE input (same shape)."""
    return lf_diff_tonemap_display(scene_linear_fchw, mu=mu)


def lf_diff_tonemap_l1(hdr: Tensor, hdr_hat: Tensor, *, mu: float = LF_DIFF_TONEMAP_MU_DEFAULT) -> Tensor:
    """Mean absolute error in LF-Diff tonemapped space (Eq. 11, first term ``||T(H)-T(H_pred)||_1`` as mean).

    Both tensors should be non-negative scene-linear RGB (same shape). This is only the **tonemapped
    L1** piece; the VGG perceptual term in Eq. 11 is left to the trainer.
    """
    return (lf_diff_tonemap_display(hdr, mu=mu) - lf_diff_tonemap_display(hdr_hat, mu=mu)).abs().mean()


def lf_diff_concat_linear_tonemap(hdr: Tensor, *, mu: float = LF_DIFF_TONEMAP_MU_DEFAULT) -> Tensor:
    """LF-Diff LPENet-style channel stack ``Concat(H, T(H))`` (Eq. 6, before ``PixelUnshuffle``).

    ``hdr`` must be non-negative linear RGB: shape ``[C, H, W]`` or ``[F, C, H, W]``. Returns the same
    spatial layout with ``2 * C`` channels (``torch.cat`` on the channel axis).
    """
    t = lf_diff_tonemap_display(hdr, mu=mu)
    if hdr.ndim == 3:
        return torch.cat([hdr, t], dim=0)
    if hdr.ndim == 4:
        return torch.cat([hdr, t], dim=1)
    raise ValueError(f"lf_diff_concat_linear_tonemap: expected 3D or 4D tensor, got {hdr.ndim}D shape {tuple(hdr.shape)}")


def lf_diff_feature_split_low_high(fmaps: Tensor, k: int = 4) -> tuple[Tensor, Tensor]:
    """LF-Diff FRM-style low / high split (Eq. 7–8) on spatial feature maps.

    ``fmaps`` is ``[C, H, W]`` or ``[B, C, H, W]``. Returns ``(low_upsampled, high)`` where
    ``low = AvgPool(f, k)`` and ``high = f - Upsample(low)``. Default ``k=4`` matches PIM in Sec. 5.1;
    use ``k=2`` for the FRM branch kernel reported there.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    batched = fmaps.ndim == 4
    x = fmaps.unsqueeze(0) if not batched else fmaps
    low = F.avg_pool2d(x, kernel_size=k, stride=k)
    low_up = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
    high = x - low_up
    if not batched:
        return low_up.squeeze(0), high.squeeze(0)
    return low_up, high


def lf_diff_lpr_l1(z_hat: Tensor, z: Tensor) -> Tensor:
    """Mean L1 between predicted and target LPR (LF-Diff Eq. 15, ``||z_hat - z||_1`` as mean over elements)."""
    return (z_hat - z).abs().mean()


def lf_diff_meta_block(*, mu: float = LF_DIFF_TONEMAP_MU_DEFAULT) -> dict[str, Any]:
    """LF-Diff provenance for ``hdr_meta`` JSON (tonemap only; LPR/DHRNet are trainer-side)."""
    return {
        "lf_diff": {
            "arxiv_id": "2404.00849",
            "tonemap_mu": mu,
            "tonemap": "log(1+mu*x)/log(1+mu)",
            "paper_impl": {
                "ddim_T": 200,
                "ddim_S_train_infer": 10,
                "L_r_vgg_lambda": 0.01,
                "lpenet_pixelunshuffle_factor": 4,
                "L_r_tonemap": "same_T_as_lpenet_eq6",
                "train_patch_hw": 128,
                "train_patch_stride": 64,
                "train_batch_size": 64,
                "dhrnet_channel_C": 60,
                "dhrnet_Ni_per_level": [3, 3, 3],
                "pim_avgpool_k": 4,
                "frm_avgpool_k": 2,
                "denoiser_unet_blocks_per_level": [2, 2, 2],
                "L_diff_prior_l1": "eq15_second_term_mean",
            },
            "training_note": "DM on compact LPR; joint-train DM+DHRNet; infer S >= train S (paper Tab. 4). Multi-exp Xi=[Li,Hi] aligns with bracket+merge stacks in this repo.",
        }
    }


def pq_eotf_bt2100(N: Tensor) -> Tensor:
    """ITU-R BT.2100 PQ EOTF: non-linear signal ``N`` in ``[0, 1]`` to linear display light.

    Returns values in ``[0, 10000]`` (nominal cd/m² scale per channel, as commonly applied).
    """
    m1 = 2610.0 / 16384.0
    m2 = (2523.0 / 4096.0) * 128.0
    c1 = 3424.0 / 4096.0
    c2 = (2413.0 / 4096.0) * 32.0
    c3 = (2392.0 / 4096.0) * 32.0
    x = N.clamp(0.0, 1.0)
    xp = x.pow(1.0 / m2)
    num = (xp - c1).clamp(min=0.0)
    den = (c2 - c3 * xp).clamp(min=1e-12)
    return (num / den).pow(1.0 / m1) * 10000.0


def hlg_oetf_inverse_bt2100(Y: Tensor) -> Tensor:
    """Inverse HLG OETF (non-linear ``Y`` in ``[0, 1]`` to scene-linear relative ``[0, 1]``)."""
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    return torch.where(Y <= 0.5, Y * Y / 3.0, (((Y - c).clamp(min=-20.0) / a).exp() + b) / 12.0)


def srgb_to_linear(x: Tensor) -> Tensor:
    """sRGB EOTF (roughly BT.709 for SDR web/video). ``x`` in ``[0, 1]``."""
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055).clamp(min=0.0).pow(2.4),
    )


def _color_trc_name(stream: Any) -> str:
    ctx = stream.codec_context
    ct = getattr(ctx, "color_transfer", None)
    if ct is None:
        return "unknown"
    name = getattr(ct, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(ct).lower()


def _resolve_transfer_mode(stream: Any, hdr_transfer: str) -> str:
    ht = hdr_transfer.strip().lower()
    if ht != "auto":
        if ht not in ("pq", "hlg", "srgb", "linear"):
            raise ValueError(
                f"Unknown hdr_transfer {hdr_transfer!r}; expected auto, pq, hlg, srgb, or linear"
            )
        return ht
    name = _color_trc_name(stream)
    if "2084" in name or "smpte2084" in name:
        return "pq"
    if "arib" in name or "hlg" in name or "std-b67" in name:
        return "hlg"
    return "srgb"


def _frame_to_rgb01(frame: av.VideoFrame, colorspace_ctx: Any | None = None) -> Tensor:
    """``[C, H, W]`` float32 in ``[0, 1]`` via FFmpeg-style YUV420 when the frame is planar YUV."""
    from ltx_trainer.yuv_colorspace import av_frame_to_rgb01

    rgb, _ = av_frame_to_rgb01(frame, colorspace_ctx)
    return rgb


def _to_scene_linear(rgb01: Tensor, transfer_mode: str) -> Tensor:
    if transfer_mode == "pq":
        lin = pq_eotf_bt2100(rgb01) / 10000.0
        return lin.clamp(min=0.0)
    if transfer_mode == "hlg":
        return hlg_oetf_inverse_bt2100(rgb01).clamp(min=0.0)
    if transfer_mode == "linear":
        return rgb01.clamp(min=0.0)
    # srgb / unknown: treat as sRGB-like display encoding
    return srgb_to_linear(rgb01.clamp(0.0, 1.0))


def image_rgb01_chw_to_scene_linear_fchw(
    rgb01_chw: Tensor,
    *,
    hdr_transfer: str = "auto",
) -> tuple[Tensor, dict[str, Any]]:
    """Convert one loaded image in display/code space to scene-linear ``[1, 3, H, W]``.

    ``rgb01_chw`` is float ``[3, H, W]`` in ``[0, 1]`` (typical ``to_tensor`` after ``open_image_as_srgb``).
    Returns ``(linear_fchw, hdr_meta)`` where ``linear_fchw`` is float32 scene-linear and ``hdr_meta``
    matches the keys produced by :func:`read_video_hdr_float32` so the same JSON merge / pack path applies.

    Stills have no container color metadata; ``hdr_transfer="auto"`` resolves to ``"srgb"``. Use
    ``"pq"`` / ``"hlg"`` / ``"linear"`` when file values are known to follow that transfer.
    """
    if rgb01_chw.ndim != 3 or rgb01_chw.shape[0] != 3:
        raise ValueError(f"Expected rgb01_chw [3,H,W], got {tuple(rgb01_chw.shape)}")
    ht = hdr_transfer.strip().lower()
    if ht == "auto":
        resolved = "srgb"
    elif ht in ("pq", "hlg", "srgb", "linear"):
        resolved = ht
    else:
        raise ValueError(f"Unsupported hdr_transfer for stills: {hdr_transfer!r}")
    meta: dict[str, Any] = {
        "hdr_transfer_requested": hdr_transfer,
        "hdr_transfer_resolved": resolved,
        "color_trc": resolved,
        "color_primaries": "unknown",
        "color_space": "unknown",
        "still_image_ingest": True,
    }
    lin = _to_scene_linear(rgb01_chw.detach().float(), resolved)
    return lin.unsqueeze(0), meta


def _read_video_hdr_via_ffmpeg(
    path: Path,
    *,
    max_frames: int | None,
    hdr_transfer: str,
) -> tuple[Tensor, float, dict[str, Any]]:
    """HDR-ish decode for QuickTime/MOV when PyAV is unreliable (treats ffmpeg RGB as display-coded)."""
    from ltx_trainer.ffmpeg_io import probe_media
    from ltx_trainer.video_utils import _read_video_ffmpeg

    pr = probe_media(path)
    fps = float(pr.video.avg_fps if pr.video else 24.0)
    ht = hdr_transfer.strip().lower()
    resolved = _resolve_transfer_mode_from_probe(pr, hdr_transfer) if ht == "auto" else ht
    if resolved not in ("pq", "hlg", "srgb", "linear"):
        resolved = "srgb"

    rgb_fchw, _ = _read_video_ffmpeg(path, max_frames)
    frames = [_to_scene_linear(rgb_fchw[f], resolved) for f in range(rgb_fchw.shape[0])]
    if not frames:
        raise RuntimeError(f"No video frames decoded from {path}")

    meta: dict[str, Any] = {
        "hdr_transfer_requested": hdr_transfer,
        "hdr_transfer_resolved": resolved,
        "decode_backend": "ffmpeg",
        "container_format": pr.format_name,
        "still_image_ingest": False,
    }
    if pr.video:
        meta["color_trc"] = pr.video.color_transfer or resolved
        meta["color_primaries"] = pr.video.color_primaries or "unknown"
        meta["color_space"] = pr.video.color_space or "unknown"
    try:
        from ltx_trainer.ffmpeg_io import colorspace_context_from_probe

        meta.update(colorspace_context_from_probe(pr))
    except Exception:
        pass
    return torch.stack(frames, dim=0), fps, meta


def _resolve_transfer_mode_from_probe(pr: Any, hdr_transfer: str) -> str:
    ht = hdr_transfer.strip().lower()
    if ht != "auto":
        return ht
    trc = (pr.video.color_transfer or "").lower() if pr.video else ""
    if "2084" in trc or "smpte2084" in trc:
        return "pq"
    if "arib" in trc or "hlg" in trc or "std-b67" in trc:
        return "hlg"
    return "srgb"


def read_video_hdr_float32(
    video_path: str | Path,
    max_frames: int | None = None,
    hdr_transfer: str = "auto",
) -> tuple[Tensor, float, dict[str, Any]]:
    """Decode video to scene-linear RGB float32 ``[F, C, H, W]``.

    Args:
        video_path: Path to a video file.
        max_frames: Maximum frames to read (``None`` = all).
        hdr_transfer: ``"auto"`` (inspect stream), ``"pq"``, ``"hlg"``, ``"srgb"``, or ``"linear"``.

    Returns:
        ``(frames, fps, meta)`` where ``frames`` is ``[F, C, H, W]`` float32, ``meta`` includes
        color hints and the resolved transfer used for linearization.
    """
    ht_req = hdr_transfer.strip().lower()
    if ht_req not in ("auto", "pq", "hlg", "srgb", "linear"):
        raise ValueError(
            f"Unknown hdr_transfer {hdr_transfer!r}; expected auto, pq, hlg, srgb, or linear"
        )
    from ltx_trainer import logger
    from ltx_trainer.video_utils import _use_ffmpeg_decode

    path = Path(video_path)
    if _use_ffmpeg_decode(path):
        return _read_video_hdr_via_ffmpeg(path, max_frames=max_frames, hdr_transfer=hdr_transfer)

    try:
        return _read_video_hdr_pyav(path, max_frames=max_frames, hdr_transfer=hdr_transfer)
    except Exception as exc:
        logger.debug("PyAV HDR decode failed for %s (%s); falling back to ffmpeg", path, exc)
        return _read_video_hdr_via_ffmpeg(path, max_frames=max_frames, hdr_transfer=hdr_transfer)


def _read_video_hdr_pyav(
    path: Path,
    *,
    max_frames: int | None,
    hdr_transfer: str,
) -> tuple[Tensor, float, dict[str, Any]]:
    from ltx_trainer.dav1d_decode import decode_info_to_meta, iter_video_frames, open_video_container
    from ltx_trainer.yuv_colorspace import colorspace_context_from_av_stream

    with open_video_container(path) as container:
        vstream = container.streams.video[0]
        fps = float(vstream.average_rate or vstream.base_rate or 24)
        resolved = _resolve_transfer_mode(vstream, hdr_transfer)
        cs_ctx = colorspace_context_from_av_stream(vstream)
        frame_iter, decode_info = iter_video_frames(container, max_frames=max_frames)
        meta: dict[str, Any] = {
            "hdr_transfer_requested": hdr_transfer,
            "hdr_transfer_resolved": resolved,
            "color_trc": _color_trc_name(vstream),
            "color_primaries": str(getattr(vstream.codec_context, "color_primaries", None)),
            "color_space": str(getattr(vstream.codec_context, "colorspace", None)),
            **cs_ctx.to_meta(),
            **decode_info_to_meta(decode_info),
        }
        try:
            from ltx_trainer.ffmpeg_io import colorspace_context_from_probe, probe_media

            meta.update(colorspace_context_from_probe(probe_media(path)))
        except Exception:
            pass

        frames: list[Tensor] = []
        for frame in frame_iter:
            rgb01 = _frame_to_rgb01(frame, cs_ctx)
            frames.append(_to_scene_linear(rgb01, resolved))

    if not frames:
        raise RuntimeError(f"No video frames decoded from {path}")

    stacked = torch.stack(frames, dim=0)
    return stacked, fps, meta


def reinhard_tonemap(linear_rgb: Tensor) -> Tensor:
    """Simple Reinhard tone-mapper for feeding HDR-ish linear RGB into an SDR-trained VAE."""
    x = linear_rgb.clamp(min=0.0)
    return x / (1.0 + x)


HDR_META_PACK_BYTES = 2048


def hdr_meta_dict_to_padded_u8(meta: dict[str, Any]) -> Tensor:
    """Serialize ``meta`` to a fixed-length ``uint8`` vector for ``DataLoader`` collation."""
    raw = json.dumps(meta, sort_keys=True).encode("utf-8")
    if len(raw) > HDR_META_PACK_BYTES:
        raise ValueError(
            f"HDR metadata JSON is {len(raw)} bytes (limit {HDR_META_PACK_BYTES}); increase HDR_META_PACK_BYTES if needed."
        )
    out = torch.zeros(HDR_META_PACK_BYTES, dtype=torch.uint8)
    out[: len(raw)] = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
    return out


def padded_u8_to_hdr_meta_dict(buf: Tensor) -> dict[str, Any]:
    """Inverse of :func:`hdr_meta_dict_to_padded_u8`."""
    b = buf.detach().cpu().numpy().tobytes().rstrip(b"\x00")
    return json.loads(b.decode("utf-8"))


def parse_ev_bracket_spec(spec: str) -> tuple[float, float, float]:
    """Parse ``"ev_min:ev_max:step"`` (e.g. ``"-7:5:1"``) for synthetic exposure stacks."""
    parts = spec.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f'Expected ev bracket "min:max:step", got {spec!r}')
    return float(parts[0]), float(parts[1]), float(parts[2])


def ev_list_arange(ev_min: float, ev_max: float, ev_step: float) -> list[float]:
    """Inclusive EV list from ``ev_min`` to ``ev_max`` with positive ``ev_step``."""
    if ev_step <= 0:
        raise ValueError("ev_step must be positive")
    out: list[float] = []
    e = ev_min
    # Guard float drift
    n = 0
    while e <= ev_max + 1e-6 * max(1.0, abs(ev_max)) and n < 4096:
        out.append(round(e, 6))
        e += ev_step
        n += 1
    return out


def synthetic_gamma_ldr_stack_from_linear_hdr(
    linear_cfhw: Tensor,
    ev_min: float,
    ev_max: float,
    ev_step: float,
    *,
    gamma: float = 2.2,
    normalize: str = "p999",
) -> tuple[Tensor, list[float]]:
    """Build a γ-encoded synthetic LDR stack in ``[0, 1]`` from scene-linear RGB (LatentHDR App. A).

    Args:
        linear_cfhw: Scene-linear RGB ``[C, F, H, W]``, non-negative.
        ev_min, ev_max, ev_step: EV range in log2 exposure units (same convention as the paper).
        gamma: Display gamma used after clipping (paper uses 2.2).
        normalize: ``"p999"`` scales linear HDR by the 99.9th percentile (finite clip analog);
            ``"max"`` divides by the tensor maximum.

    Returns:
        ``(stack, ev_list)`` with ``stack`` shape ``[N, C, F, H, W]`` float32 in display space
        (γ-encoded, pre-quantization).
    """
    if linear_cfhw.ndim != 4:
        raise ValueError(f"linear_cfhw must be [C,F,H,W], got {linear_cfhw.shape}")
    x = linear_cfhw.detach().float().clamp(min=0.0)
    flat = x.reshape(-1)
    if normalize == "max":
        peak = flat.max().clamp(min=1e-8)
    elif normalize == "p999":
        peak = torch.quantile(flat, 0.999).clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown normalize mode {normalize!r}")
    x_hdr = x / peak
    ev_list = ev_list_arange(ev_min, ev_max, ev_step)
    if not ev_list:
        raise ValueError("Empty EV list; check ev_min, ev_max, ev_step")
    outs: list[Tensor] = []
    for e in ev_list:
        scale = 2.0 ** float(e)
        x_e = (x_hdr * scale).clamp(0.0, 1.0)
        y_e = x_e.pow(1.0 / gamma)
        outs.append(y_e)
    return torch.stack(outs, dim=0), ev_list


def merge_log_domain_radiance(
    ldrs_display: Tensor,
    ev_list: list[float],
    *,
    gamma: float = 2.2,
    tau_lo: float = 0.02,
    tau_hi: float = 0.98,
    eps: float = 1e-8,
) -> Tensor:
    """Fuse a synthetic exposure stack into a single linear radiance map (LatentHDR Sec. 3.5, Eq. 9).

    Args:
        ldrs_display: Stack ``[N, C, F, H, W]`` in ``[0, 1]``, γ-encoded (same domain as
            :func:`synthetic_gamma_ldr_stack_from_linear_hdr` outputs).
        ev_list: Length ``N`` list of EV offsets matching dim 0 of ``ldrs_display``.

    Returns:
        Linear radiance estimate ``[C, F, H, W]`` (relative units, consistent with the
        ``2**ev`` normalization used inside this function).
    """
    if ldrs_display.ndim != 5:
        raise ValueError(f"Expected [N,C,F,H,W], got {ldrs_display.shape}")
    n = ldrs_display.shape[0]
    if len(ev_list) != n:
        raise ValueError(f"ev_list length {len(ev_list)} != N={n}")
    device, dtype = ldrs_display.device, ldrs_display.dtype
    ev_t = torch.tensor(ev_list, device=device, dtype=dtype).view(n, 1, 1, 1, 1)
    ldr = ldrs_display.clamp(0.0, 1.0)
    x_lin = ldr.pow(gamma)
    radiance = x_lin / (2.0 ** ev_t)

    m = ldr.mean(dim=1)
    tri = (4.0 * torch.minimum(m, 1.0 - m)).clamp(min=0.0, max=1.0)
    valid = ((ldr > tau_lo) & (ldr < tau_hi)).all(dim=1)
    w = tri * valid.float()

    log_r = (w.unsqueeze(1) * torch.log(radiance.clamp(min=eps))).sum(dim=0)
    den = w.sum(dim=0).unsqueeze(0).clamp(min=eps)
    return torch.exp(log_r / den).to(dtype=ldrs_display.dtype)


def latenthdr_meta_block(
    *,
    ev_recipe: tuple[float, float, float] = (-7.0, 5.0, 1.0),
    save_ldr_stack: bool = False,
    ev_spec_saved: str | None = None,
) -> dict[str, Any]:
    """Compact metadata block citing LatentHDR conventions (for JSON / ``hdr_meta``)."""
    ev_min, ev_max, ev_step = ev_recipe
    return {
        "latenthdr": {
            "arxiv_id": "2605.11115",
            "scene_anchor_key": "hdr_latent",
            "vae_latent_role": "posterior_mean_normalized",
            "synthetic_bracket_recipe": {"ev_min": ev_min, "ev_max": ev_max, "ev_step": ev_step, "gamma": 2.2},
            "save_ldr_ev_stack": save_ldr_stack,
            "ev_spec_saved": ev_spec_saved,
        }
    }
