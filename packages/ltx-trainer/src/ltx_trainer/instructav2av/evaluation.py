"""Evaluation demos and framework card."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.instructav2av.benchmarks import benchmarks_bundle
from ltx_trainer.instructav2av.config import InstructAV2AVConfig
from ltx_trainer.instructav2av.data_pipeline import data_engine_card, pipeline_stages, verification_scorecard
from ltx_trainer.instructav2av.dataset import dataset_card, example_sample
from ltx_trainer.instructav2av.edit import instructav2av_edit
from ltx_trainer.instructav2av.siga import SIGAModule
from ltx_trainer.instructav2av.train_step import TrainingStage, instructav2av_training_step, training_schedule


def knowledge_card() -> dict[str, Any]:
    cfg = InstructAV2AVConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "authors": "Haojie Zheng, Yixin Yang, Siqi Yang, Shuchen Weng, Boxin Shi",
        "website": cfg.website,
        "method": (
            "Dual-stream flow matching on Ovi-style backbone; channel-concat source latents; "
            "SIGA for mask-free instruction following; two-stage training"
        ),
        "dataset": dataset_card(),
        "ltx_hook": cfg.ltx_hook,
    }


def framework_card() -> dict[str, Any]:
    cfg = InstructAV2AVConfig()
    return {
        "name": cfg.name,
        "config": {
            "lambda_video": cfg.lambda_video,
            "lambda_audio": cfg.lambda_audio,
            "use_source_concat": cfg.use_source_concat,
            "use_siga": cfg.use_siga,
            "two_stage_training": cfg.two_stage_training,
        },
        "data_pipeline": {
            "stages": pipeline_stages(cfg),
            "verification": verification_scorecard(),
            "engine": data_engine_card(),
        },
        "training": training_schedule(cfg),
        "benchmarks": benchmarks_bundle(),
        "knowledge": knowledge_card(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    torch.manual_seed(seed)
    cfg = InstructAV2AVConfig()
    siga_dim = 32
    token_dim = 64
    b, c, h, w = 1, 4, 8, 8

    zs_v = torch.randn(b, c, h, w)
    zs_a = torch.randn(b, c, h, w)
    z1_v = torch.randn(b, c, h, w)
    z1_a = torch.randn(b, c, h, w)
    inst = torch.randn(b, token_dim)

    siga = SIGAModule(siga_dim)
    fh = torch.randn(b, 16, siga_dim)
    fx = torch.randn(b, 16, siga_dim)
    fc = torch.randn(b, 16, siga_dim)
    fused, gate_mean = siga(fh, fx, fc)

    joint = instructav2av_training_step(z1_v, z1_a, zs_v, zs_a, inst, stage=TrainingStage.JOINT, cfg=cfg)
    z_out_v, z_out_a = instructav2av_edit(zs_v, zs_a, inst, steps=4, cfg=cfg)

    sample = example_sample()
    return {
        "siga": {"fused_shape": list(fused.shape), "gate_mean": float(gate_mean.detach().mean())},
        "training": {"loss": float(joint["loss"].detach().item()), "stage": joint["stage"]},
        "inference": {"z_out_v": list(z_out_v.shape), "z_out_a": list(z_out_a.shape)},
        "example_instruction": sample.instruction[:80] + "…",
        "table1_insave_fvd": benchmarks_bundle()["table1_insave"]["InstructAV2AV"]["FVD"],
    }
