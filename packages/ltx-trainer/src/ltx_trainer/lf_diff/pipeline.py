"""LF-Diff pipeline + evaluation demos."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.lf_diff.benchmarks import benchmarks_bundle
from ltx_trainer.lf_diff.model import LfDiff, LfDiffConfig
from ltx_trainer.lf_diff.paper import framework_card
from ltx_trainer.lf_diff.synthetic import synthesize_exposure_bracket, synthesize_hdr_scene
from ltx_trainer.lf_diff.tonemap import tonemap_l1


def pipeline_demo(*, device: str | torch.device = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    hdr = synthesize_hdr_scene(32, 32).to(dev)
    brackets = [b.to(dev) for b in synthesize_exposure_bracket(hdr)]
    model = LfDiff(LfDiffConfig()).to(dev)
    model.eval()
    with torch.no_grad():
        pred = model.infer(brackets, steps=3)
        tm_l1 = float(tonemap_l1(pred, hdr))
    return {
        "framework": framework_card(),
        "bracket_count": len(brackets),
        "output_shape": list(pred.shape),
        "tonemap_l1": tm_l1,
    }


def evaluation_demo(*, device: str | torch.device = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    model = LfDiff(LfDiffConfig()).to(dev)
    model.train()
    hdr = synthesize_hdr_scene(32, 32).to(dev)
    brackets = [b.to(dev) for b in synthesize_exposure_bracket(hdr)]
    loss, metrics = model.training_step(brackets, hdr)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    return {"loss": float(loss.detach()), "grad_norm": grad_norm, **metrics}


def train_step(
    model: LfDiff,
    brackets: list[torch.Tensor],
    hdr: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    loss, metrics = model.training_step(brackets, hdr)
    loss.backward()
    if optimizer is not None:
        optimizer.step()
    metrics["loss"] = float(loss.detach())
    return metrics
