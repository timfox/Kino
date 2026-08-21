"""Training / eval demo steps."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ltx_trainer.gsparc.config import GSpaRCConfig
from ltx_trainer.gsparc.densify import densify_step_report
from ltx_trainer.gsparc.downstream import pilot_free_coverage
from ltx_trainer.gsparc.gsparc_net import GSpaRCStub
from ltx_trainer.gsparc.losses import normalize_confidence, spectrum_loss
from ltx_trainer.gsparc.synthetic import synthetic_channel, synthetic_rx_position, synthetic_spectrum


def evaluation_demo_run(cfg: GSpaRCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GSpaRCConfig(spectrum_height=32, spectrum_width=64, num_gaussians=32)
    model = GSpaRCStub(cfg)
    x_rx = synthetic_rx_position()
    z_gt = synthetic_spectrum(cfg)
    out = model(x_rx)
    losses = model.training_losses(x_rx, target_spectrum=z_gt)

    cfg_ch = GSpaRCConfig(objective="channel", num_gaussians=32)
    model_ch = GSpaRCStub(cfg_ch)
    losses_ch = model_ch.training_losses(
        x_rx, target_channel=synthetic_channel()
    )

    return {
        "spectrum_shape": list(out["spectrum"].shape),
        "channel": [
            float(out["channel"][0].detach()),
            float(out["channel"][1].detach()),
        ],
        "confidence_raw": float(out["confidence_raw"].detach()),
        "loss_spectrum": float(losses["loss"].detach()),
        "loss_channel": float(losses_ch["loss"].detach()),
        "render_stub_ms_order": "sub-ms to low-ms (paper: 0.8–3.8 ms on RTX 2080 Ti)",
    }


def ablation_distance(cfg: GSpaRCConfig | None = None) -> dict[str, float]:
    """Sec. 7.1 — spectrum loss with vs without 1/d attenuation."""
    cfg = cfg or GSpaRCConfig(spectrum_height=24, spectrum_width=48, num_gaussians=16)
    x_rx = synthetic_rx_position()
    z_gt = synthetic_spectrum(cfg)
    on = GSpaRCStub(replace(cfg, use_distance_attenuation=True))
    off = GSpaRCStub(replace(cfg, use_distance_attenuation=False))
    with torch.no_grad():
        loss_on = float(
            spectrum_loss(on(x_rx)["spectrum"], z_gt, lambda_l1=cfg.lambda_l1_ssim).detach()
        )
        loss_off = float(
            spectrum_loss(off(x_rx)["spectrum"], z_gt, lambda_l1=cfg.lambda_l1_ssim).detach()
        )
    return {"loss_with_1_over_d": loss_on, "loss_without_1_over_d": loss_off}


def pilot_free_demo(n_positions: int = 800) -> dict[str, Any]:
    """Synthetic confidence spread → pilot-free fraction (Fig. 2)."""
    raw = torch.linspace(1.0, 2.5, n_positions)
    norm = normalize_confidence(raw)
    return pilot_free_coverage(norm)


def densify_report(cfg: GSpaRCConfig | None = None) -> dict[str, int]:
    cfg = cfg or GSpaRCConfig(num_gaussians=32)
    model = GSpaRCStub(cfg)
    g = cfg.num_gaussians
    grad = torch.rand(g) * 0.001
    return densify_step_report(
        grad,
        model.gaussian_opacity().detach(),
        torch.exp(model.log_scale).max(dim=1).values.detach(),
    )


def train_step(cfg: GSpaRCConfig | None = None) -> dict[str, float]:
    cfg = cfg or GSpaRCConfig(spectrum_height=24, spectrum_width=48, num_gaussians=16)
    model = GSpaRCStub(cfg)
    x_rx = synthetic_rx_position()
    losses = model.training_losses(x_rx, target_spectrum=synthetic_spectrum(cfg))
    losses["loss"].backward()
    return {
        "loss": float(losses["loss"].detach()),
        "task_loss": float(losses["task_loss"].detach()),
    }
