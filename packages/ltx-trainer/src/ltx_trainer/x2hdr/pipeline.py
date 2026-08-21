"""X2HDR pipeline + evaluation demos."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.x2hdr.benchmarks import benchmarks_bundle
from ltx_trainer.x2hdr.model import X2Hdr, X2HdrConfig
from ltx_trainer.x2hdr.paper import framework_card
from ltx_trainer.x2hdr.pu21_codec import scene_linear_to_vae_pixels
from ltx_trainer.x2hdr.synthetic import synthesize_hdr_scene, synthesize_sdr_from_hdr


def pipeline_demo(*, device: str | torch.device = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    hdr = synthesize_hdr_scene(32, 32).to(dev)
    sdr = synthesize_sdr_from_hdr(hdr).to(dev)
    model = X2Hdr(X2HdrConfig()).to(dev)
    model.eval()
    with torch.no_grad():
        pred = model.infer(sdr, steps=3)
        pu21_gt = scene_linear_to_vae_pixels(hdr)
        pu21_pred = scene_linear_to_vae_pixels(pred.clamp(min=0.0))
        pu21_l1 = float((pu21_gt - pu21_pred).abs().mean())
    return {
        "framework": framework_card(),
        "input_shape": list(sdr.shape),
        "output_shape": list(pred.shape),
        "pu21_l1": pu21_l1,
        "denoise_steps": 3,
    }


def evaluation_demo(*, device: str | torch.device = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    model = X2Hdr(X2HdrConfig()).to(dev)
    model.train()
    hdr = synthesize_hdr_scene(32, 32).to(dev)
    sdr = synthesize_sdr_from_hdr(hdr).to(dev)
    loss, metrics = model.training_step(sdr, hdr)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    return {"loss": float(loss.detach()), "grad_norm": grad_norm, **metrics}


def train_step(
    model: X2Hdr,
    sdr: torch.Tensor,
    hdr: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    loss, metrics = model.training_step(sdr, hdr)
    loss.backward()
    if optimizer is not None:
        optimizer.step()
    metrics["loss"] = float(loss.detach())
    return metrics
