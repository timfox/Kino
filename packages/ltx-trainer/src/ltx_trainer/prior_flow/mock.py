"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.prior_flow.benchmarks import TABLE6_SOTA, benchmarks_bundle
from ltx_trainer.prior_flow.config import PAPER_ARXIV, PriorFlowConfig
from ltx_trainer.prior_flow.datasets import datasets_card
from ltx_trainer.prior_flow.paper import framework_card
from ltx_trainer.prior_flow.pipeline import (
    ablation_modules,
    distortion_demo,
    evaluation_demo_run,
    orthogonal_demo,
    train_step,
)
from ltx_trainer.prior_flow.prior_flow_net import PriorFlowStub
from ltx_trainer.prior_flow.synthetic import synthetic_erp_pair


def evaluation_smoke() -> dict[str, Any]:
    cfg = PriorFlowConfig(height=64, width=128, num_iterations=3)
    model = PriorFlowStub(cfg)
    f1, f2 = synthetic_erp_pair(cfg)
    out = model(f1, f2)
    ours = TABLE6_SOTA["PriOr_RAFT"]
    slof = TABLE6_SOTA["SLOF_RAFT"]

    return {
        "package": "prior_flow",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "mpf_eft_epe": ours["mpf_eft_epe"],
        "beats_slof_mpf_eft": ours["mpf_eft_epe"] < slof["mpf_eft_epe"],
        "flowscape_epe": ours["flowscape_all_epe"],
        "flow_p_shape": list(out["flow_primitive"].shape),
        "train": train_step(cfg),
        "demo": evaluation_demo_run(cfg),
        "ablation": ablation_modules(cfg),
        "distortion": distortion_demo(cfg.height, cfg.width),
        "orthogonal": orthogonal_demo(cfg),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "table6_methods": list(TABLE6_SOTA.keys()),
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
