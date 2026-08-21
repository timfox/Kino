"""Framework card, demos, and evaluation smoke (arXiv:2606.05433)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.zk_training.config import ZkTrainingConfig
from ltx_trainer.zk_training.metrics import (
    open_problems_catalog,
    prior_paradigm_overhead_estimate,
    recommended_sample_k,
    table1_zkml_comparison,
    table10_cost_summary,
    table2_training_overhead,
)
from ltx_trainer.zk_training.protocol import run_toy_mlp_protocol
from ltx_trainer.zk_training.sampling import table6_sample_counts


def framework_card(cfg: ZkTrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ZkTrainingConfig()
    return {
        "name": "ZK Frontier Training Verification",
        "paper": cfg.paper_arxiv,
        "task": "Governance-enforceable zkVM proof of faithful dense pre-training",
        "trust_anchors": [
            "Pre-committed training specification (private arch_spec)",
            "Inter-node network TAP / attested SmartNIC observations",
            "On-the-fly Merkle commitments of intermediate computation",
        ],
        "proof_types": ["genesis (P.1)", "in-training step (P.2)", "ex-ante attestation (P.3)"],
        "precompiles": [p.name for p in cfg.precompiles],
        "regulatory_thresholds": {
            "eu_ai_act_flops": cfg.thresholds.eu_ai_act_flops,
            "llama31_405b_flops": cfg.thresholds.llama31_405b_flops,
        },
        "deployment_poc_months": cfg.deployment_months_poc,
        "aggregate_proof_kb": cfg.aggregate_proof_kb,
        "sample_k_default": recommended_sample_k(cfg),
        "upstream": "https://gpaipolicylab.org/verification",
    }


def paper_limitations() -> list[str]:
    return [
        "RISC Zero / cuDSS-class zkVM stack not bundled — numpy toy MLP stands in for GPU re-run + STARK.",
        "Proof checker verifies hint-and-verify semantics, not full 405B-layer constraint budget.",
        "Network TAP Tier 1 vs SmartNIC Tier 2 trust models are documented but not simulated on wire.",
        "PAC-style learnability (Appendix G.6) is motivational; protocol proves execution-conformance only.",
        "Sparse spot-check soundness is detection-grade; universal coverage needs V5-IVC upgrade path.",
    ]


def evaluation_demo(cfg: ZkTrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ZkTrainingConfig()
    run = run_toy_mlp_protocol(cfg=cfg)
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "table1_rows": len(table1_zkml_comparison()),
        "table6_rows": len(table6_sample_counts()),
        "open_problems": open_problems_catalog(),
        "prior_vs_target": prior_paradigm_overhead_estimate(),
        "table2": table2_training_overhead(cfg),
        "table10": table10_cost_summary(cfg),
        "toy_protocol": {
            "genesis_pass": run.genesis_pass,
            "ex_ante_ok": run.ex_ante_ok,
            "steps": len(run.step_records),
            "proof_constraints": run.proof_constraints,
            "h_commit_hex": run.h_commit.hex()[:16],
        },
    }


def evaluation_smoke(cfg: ZkTrainingConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    toy = demo["toy_protocol"]
    assert toy["genesis_pass"]
    assert toy["ex_ante_ok"]
    assert demo["table1_rows"] >= 5
    assert len(demo["open_problems"]) == 13
    target = next(r for r in table1_zkml_comparison() if r["scope"].startswith("Target"))
    assert target["proves_hw_exec"] == "Yes"
    assert recommended_sample_k(cfg or ZkTrainingConfig()) == 4605
    return {"status": "ok", "paper": (cfg or ZkTrainingConfig()).paper_arxiv, "demo_keys": list(demo.keys())}
