"""HDR-to-display encodings for manifold alignment analysis (Fig. 4, Table 1)."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.hdr_ingest import linear_scene_to_pu21_display, prepare_scene_linear_for_vae
from ltx_trainer.lumivid.logc3_codec import (
    scene_linear_to_vae_pixels,
    vae_denormalize,
    vae_normalize,
    vae_pixels_to_scene_linear,
)


class HdrEncoding(str, Enum):
    LOGC3 = "logc3"
    PQ = "pq"
    HLG = "hlg"
    ACES = "aces"


def encode_hdr(scene_linear: Tensor, encoding: HdrEncoding | str) -> Tensor:
    enc = HdrEncoding(encoding)
    x = prepare_scene_linear_for_vae(scene_linear)
    if enc == HdrEncoding.LOGC3:
        return scene_linear_to_vae_pixels(x)
    if enc == HdrEncoding.PQ:
        return linear_scene_to_pu21_display(x)
    if enc == HdrEncoding.HLG:
        a, b = 0.17883277, 0.28466892
        t = (x.clamp(0.0, 1.0) * 12.0).clamp(max=1.0)
        return torch.where(t <= 0.5, (t * t) / 3.0, (torch.exp((t - b) / a) + b) / 12.0).clamp(0.0, 1.0)
    return ((torch.log2(x.clamp(min=1e-4) * 0.18 + 1e-4) + 9.72) / 17.52).clamp(0.0, 1.0)


def decode_hdr(display: Tensor, encoding: HdrEncoding | str) -> Tensor:
    enc = HdrEncoding(encoding)
    if enc == HdrEncoding.LOGC3:
        return vae_pixels_to_scene_linear(display)
    if enc == HdrEncoding.PQ:
        from ltx_trainer.hdr_ingest import X2HDR_DEFAULT_L_PEAK_CD_M2, pu21_inverse_to_linear_abs

        peak = display.amax().clamp(min=1e-6)
        return pu21_inverse_to_linear_abs(display) * (peak / X2HDR_DEFAULT_L_PEAK_CD_M2)
    return display


def kl_histogram_proxy(samples: Tensor, reference: Tensor, bins: int = 64) -> float:
    s = samples.detach().flatten().clamp(0.0, 1.0)
    r = reference.detach().flatten().clamp(0.0, 1.0)
    hs = torch.histc(s, bins=bins, min=0.0, max=1.0) + 1e-6
    hr = torch.histc(r, bins=bins, min=0.0, max=1.0) + 1e-6
    ps = hs / hs.sum()
    pr = hr / hr.sum()
    return float((ps * (torch.log(ps) - torch.log(pr))).sum().item())


def vae_roundtrip_mse(scene_linear: Tensor, encoding: HdrEncoding | str, encoder_decoder) -> float:
    disp = encode_hdr(scene_linear, encoding)
    z = encoder_decoder.encode(vae_normalize(disp))
    recon_disp = vae_denormalize(encoder_decoder.decode(z))
    back = decode_hdr(recon_disp, encoding)
    return float(F.mse_loss(back, scene_linear.clamp(min=0.0)).item())
