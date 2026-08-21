"""CPU smoke."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.gsparc.benchmarks import benchmarks_bundle
from ltx_trainer.gsparc.config import CONFIDENCE_THRESHOLD, PAPER_ARXIV, GSpaRCConfig
from ltx_trainer.gsparc.datasets import datasets_card
from ltx_trainer.gsparc.gsparc_net import GSpaRCStub
from ltx_trainer.gsparc.losses import normalize_confidence
from ltx_trainer.gsparc.paper import framework_card
from ltx_trainer.gsparc.downstream import downstream_card
from ltx_trainer.gsparc.hyperparams import training_hyperparameters
from ltx_trainer.gsparc.pipeline import (
    ablation_distance,
    densify_report,
    evaluation_demo_run,
    pilot_free_demo,
    train_step,
)
from ltx_trainer.gsparc.synthetic import synthetic_rx_position


def evaluation_smoke() -> dict[str, Any]:
    cfg = GSpaRCConfig(spectrum_height=24, spectrum_width=48, num_gaussians=24)
    model = GSpaRCStub(cfg)
    x_rx = synthetic_rx_position()
    out = model(x_rx)
    bundle = benchmarks_bundle()
    step = train_step(cfg)

    conf_norm = normalize_confidence(
        torch.stack([out["confidence_raw"], out["confidence_raw"] * 1.1])
    )

    return {
        "package": "gsparc",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "spectrum_shape": list(out["spectrum"].shape),
        "sionna_ssim": bundle["table3_sionna_conference"]["GSpaRC"]["ssim_mean"],
        "rfid_render_ms": bundle["table4_rfid"]["GSpaRC"]["render_ms"],
        "argos_nmse": bundle["table5_argos"]["GSpaRC"]["nmse"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "pilot_free_fraction": bundle["argos_confidence"]["pilot_free_fraction"],
        "gsparc_faster_than_gsrf_render": bundle["table1_sionna"]["GSpaRC"]["render_ms"]
        < bundle["table1_sionna"]["GSRF"]["render_ms"],
        "confidence_normalized_range": [
            float(conf_norm.min().detach()),
            float(conf_norm.max().detach()),
        ],
        "train_loss": step["loss"],
        "demo": evaluation_demo_run(cfg),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "hyperparams": training_hyperparameters(),
        "downstream": downstream_card(),
        "ablation_1_over_d": ablation_distance(cfg),
        "densify_report": densify_report(cfg),
        "pilot_free_synthetic": pilot_free_demo(),
    }
