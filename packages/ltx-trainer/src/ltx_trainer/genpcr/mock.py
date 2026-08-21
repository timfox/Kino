"""CPU smoke."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.genpcr.benchmarks import benchmarks_bundle
from ltx_trainer.genpcr.config import PAPER_ARXIV, GenPcrConfig
from ltx_trainer.genpcr.coupled_denoise import couple_latents, decouple_latents
from ltx_trainer.genpcr.datasets import datasets_card
from ltx_trainer.genpcr.paper import framework_card
from ltx_trainer.genpcr.pipeline import evaluation_demo_run
from ltx_trainer.genpcr.finetune import denoising_matching_loss, forward_diffusion_stub
from ltx_trainer.genpcr.registration_stub import GenerativePcrStub
from ltx_trainer.genpcr.synthetic import synthetic_pair
from ltx_trainer.genpcr.theory import coupled_elbo_summary


def evaluation_smoke() -> dict[str, Any]:
    cfg = GenPcrConfig()
    p, q = synthetic_pair(cfg)
    out = GenerativePcrStub(cfg)(p, q)
    bundle = benchmarks_bundle()

    a = torch.randn(1, 8, 16, 16)
    b = torch.randn(1, 8, 16, 16)
    z = couple_latents(a, b)
    z_p, z_q = decouple_latents(z)

    x0 = torch.randn(1, 4, 16, 16)
    x_t, eps = forward_diffusion_stub(x0, t=100)
    loss = denoising_matching_loss(eps, eps)

    return {
        "package": "genpcr",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "desc_dim": out["desc_p"].shape[-1],
        "threedmatch_rot45": bundle["threedmatch_generative_fcgf_sd"]["rot_45_acc"],
        "dur360_predator_ir": bundle["dur360_generative"]["Predator_SD_IR"],
        "coupled_latent_ok": z_p.shape[-2] + z_q.shape[-2] == z.shape[-2],
        "finetune_loss": float(loss.item()),
        "color_pcd_dim": out["color_pcd_p"].shape[-1],
        "theory": coupled_elbo_summary()["objective"],
        "demo": evaluation_demo_run(cfg),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
    }
