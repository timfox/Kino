"""Training and evaluation pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.lumaflux.config import LumaFluxConfig
from ltx_trainer.lumaflux.losses import LumaFluxLoss, LumaFluxLossConfig
from ltx_trainer.lumaflux.metrics import TABLE1_BENCHMARKS, TABLE3_ABLATION
from ltx_trainer.lumaflux.model import LumaFlux


def train_step(
    model: LumaFlux,
    loss_fn: LumaFluxLoss,
    *,
    sdr: Tensor,
    hdr_gt: Tensor,
    t: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(sdr, t=t)
    return loss_fn(
        out.hdr,
        hdr_gt,
        spline_logits=out.spline_logits,
        spline_smooth_fn=model.rqs.spline_smoothness_loss,
    )


def ablation_configs() -> list[LumaFluxConfig]:
    return [
        LumaFluxConfig(use_pga=False, use_pcm=False, use_coupler=False, use_spectral_gating=False),
        LumaFluxConfig(use_pga=True, use_pcm=False, use_coupler=False, use_spectral_gating=False),
        LumaFluxConfig(use_pga=True, use_pcm=False, use_coupler=False, use_spectral_gating=True),
        LumaFluxConfig(use_pga=True, use_pcm=True, use_coupler=True, use_spectral_gating=True),
    ]


def paper_report() -> dict[str, object]:
    return {
        "table1": TABLE1_BENCHMARKS,
        "table3_ablation": TABLE3_ABLATION,
        "best_benchmark": "luma_eval",
    }


def save_checkpoint(model: LumaFlux, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path | str, *, device: str = "cpu") -> LumaFlux:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg = LumaFluxConfig(**ckpt.get("config", {}))
    model = LumaFlux(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()
