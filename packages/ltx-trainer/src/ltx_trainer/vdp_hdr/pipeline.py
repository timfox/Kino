"""Inference helpers and checkpoint I/O."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.vdp_hdr.fusion import FusionUNet, bracket_to_linear, fuse_bracket
from ltx_trainer.vdp_hdr.model import VdpHdr, VdpHdrConfig


def load_vdp_hdr_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> VdpHdr:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=True)
    raw = ckpt.get("config", {})
    cfg = VdpHdrConfig(**{f.name: raw[f.name] for f in dataclass_fields(VdpHdrConfig) if f.name in raw})
    model = VdpHdr(cfg)
    state = ckpt["state_dict"]
    # Fusion-only training checkpoints (train_vdp_hdr_fusion.py) omit bracket prior weights.
    if state and not any(k.startswith("fusion.") for k in state):
        model.fusion.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


@torch.inference_mode()
def recover_hdr_from_ldr(
    model: VdpHdr,
    ldr_chw: Tensor,
    *,
    external_bracket_bncHW: Tensor | None = None,
) -> Tensor:
    """Recover linear HDR ``[C,H,W]`` from γ-encoded LDR.

    When ``external_bracket_bncHW`` is set (e.g. from fine-tuned LTX), skip the photometric prior.
    """
    if ldr_chw.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(ldr_chw.shape)}")
    if external_bracket_bncHW is not None:
        bracket = external_bracket_bncHW
        if bracket.ndim == 4:
            bracket = bracket.unsqueeze(0)
        hdr = model.fuse(bracket)
        return hdr.squeeze(0)
    hdr, _ = model(ldr_chw)
    return hdr


def bracket_from_ltx_video(
    video_fchw: Tensor,
    *,
    gamma: float = 2.2,
) -> Tensor:
    """Use an LTX-generated exposure stack ``[F,C,H,W]`` as bracket ``[1,N,C,H,W]``."""
    if video_fchw.ndim != 4:
        raise ValueError(f"Expected [F,C,H,W], got {tuple(video_fchw.shape)}")
    return video_fchw.unsqueeze(0).clamp(0.0, 1.0)


def fuse_bracket_with_model(model: VdpHdr, bracket_bncHW: Tensor) -> Tensor:
    return model.fuse(bracket_bncHW)


def train_step(
    model: VdpHdr,
    *,
    ldr: Tensor | None = None,
    hdr: Tensor | None = None,
    bracket: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """One fusion-training step on synthetic or provided bracket/HDR pair."""
    from ltx_trainer.vdp_hdr.losses import FusionLoss
    from ltx_trainer.vdp_hdr.metrics import q_mae, q_psnr
    from ltx_trainer.vdp_hdr.synthetic import synthesize_hdr_pair

    loss_fn = FusionLoss()
    if hdr is None or bracket is None:
        side = 48
        if ldr is not None and ldr.ndim >= 3:
            side = int(ldr.shape[-1])
        hdr_t, _ldr_t, bracket_t = synthesize_hdr_pair(side, side)
        hdr = hdr_t.unsqueeze(0)
        bracket = bracket_t.unsqueeze(0)
    if bracket.ndim == 4:
        bracket = bracket.unsqueeze(0)
    if hdr.ndim == 3:
        hdr = hdr.unsqueeze(0)

    lin = bracket_to_linear(bracket, gamma=model.config.gamma)
    if model.fusion is None:
        pred = model.fuse(bracket)
    else:
        w = model.fusion(lin)
        pred = fuse_bracket(lin, w)
    loss, parts = loss_fn(pred, hdr)
    stats = {
        "loss_fusion": float(loss.detach()),
        "q_psnr": float(q_psnr(pred.squeeze(0).detach(), hdr.squeeze(0))),
        "q_mae": float(q_mae(pred.squeeze(0).detach(), hdr.squeeze(0))),
        **{k: float(v) for k, v in parts.items()},
    }
    return loss, stats
