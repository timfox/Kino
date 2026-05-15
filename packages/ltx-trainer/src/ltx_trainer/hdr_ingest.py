"""HDR-oriented video decode: float32 tensors and LatentHDR-aligned radiometry helpers.

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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from torch import Tensor


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
    if hdr_transfer != "auto":
        return hdr_transfer
    name = _color_trc_name(stream)
    if "2084" in name or "smpte2084" in name:
        return "pq"
    if "arib" in name or "hlg" in name or "std-b67" in name:
        return "hlg"
    return "srgb"


def _frame_to_rgb01(frame: av.VideoFrame) -> Tensor:
    """``[C, H, W]`` float32 in ``[0, 1]`` (display/code values). Prefer 16-bit RGB when available."""
    try:
        conv = frame.reformat(format="rgb48le")
        arr = conv.to_ndarray()
        t = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1).div_(65535.0)
    except Exception:
        arr = frame.to_ndarray(format="rgb24")
        t = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1).div_(255.0)
    return t


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
    path = Path(video_path)
    with av.open(str(path)) as container:
        vstream = container.streams.video[0]
        fps = float(vstream.average_rate or vstream.base_rate or 24)
        resolved = _resolve_transfer_mode(vstream, hdr_transfer)
        meta: dict[str, Any] = {
            "hdr_transfer_requested": hdr_transfer,
            "hdr_transfer_resolved": resolved,
            "color_trc": _color_trc_name(vstream),
            "color_primaries": str(getattr(vstream.codec_context, "color_primaries", None)),
            "color_space": str(getattr(vstream.codec_context, "color_space", None)),
        }

        frames: list[Tensor] = []
        for frame in container.decode(video=0):
            if max_frames is not None and len(frames) >= max_frames:
                break
            rgb01 = _frame_to_rgb01(frame)
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
