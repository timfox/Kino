"""Paper benchmark anchors — Tables 1, 2, 9–10, zk-ML comparison."""

from __future__ import annotations

from typing import Any

from ltx_trainer.zk_training.config import ZkTrainingConfig
from ltx_trainer.zk_training.sampling import sample_count


def table1_zkml_comparison() -> list[dict[str, Any]]:
    """Table 1: existing ZK-ML vs target architecture."""
    cols = ("scope", "arithmetic", "max_scale", "multi_step", "distributed", "proves_hw_exec")
    rows = [
        ("ZKML", "Finite field", "81M", "Per-sample", "No", "No"),
        ("zkLLM", "Finite field", "13B", "Single pass", "No", "No"),
        ("ezkl", "Finite field", "250K", "Per-sample", "No", "No"),
        ("VeriLoRA", "Finite field", "13B", "1 LoRA step", "No", "No"),
        ("Target (this paper)", "Native BF16", "100-1000B (est.)", "Full run", "Yes", "Yes"),
    ]
    return [dict(zip(cols, row, strict=True)) for row in rows]


def table2_training_overhead(cfg: ZkTrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ZkTrainingConfig()
    lo, hi = cfg.determinism_overhead_pct
    mlo, mhi = cfg.merkle_hash_overhead_pct
    budget = cfg.thresholds.training_budget_usd
    return {
        "determinism_pct": [lo, hi],
        "determinism_usd": [budget * lo / 100, budget * hi / 100],
        "merkle_pct": [mlo, mhi],
        "merkle_usd": [budget * mlo / 100, budget * mhi / 100],
        "weight_storage_tb": 81,
        "total_pct": [lo + mlo, hi + mhi],
        "total_usd": [budget * (lo + mlo) / 100, budget * (hi + mhi) / 100],
    }


def table10_cost_summary(cfg: ZkTrainingConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or ZkTrainingConfig()
    budget = cfg.thresholds.training_budget_usd
    return [
        {"component": "Determinism tax", "usd": [1.6e6, 8.2e6], "pct": [1.6, 8.2]},
        {"component": "Concurrent Merkle hashing", "usd": [0.5e6, 1.5e6], "pct": [0.5, 1.5]},
        {"component": "Genesis proof (reduced batch)", "usd": 1000, "pct": 0.01},
        {"component": "In-training challenges (layered mix)", "usd": 326_000, "pct": 0.33},
        {"component": "Weight storage (81 TB rolling)", "usd": 50_000, "pct": 0.1},
        {"component": "Total", "usd": [2e6, 10e6], "pct": [2, 10]},
    ]


def open_problems_catalog() -> list[dict[str, str]]:
    return [
        {"id": "OP-1", "area": "Proof foundations", "topic": "Algebraic FP GEMM verification / approximate sumcheck"},
        {"id": "OP-2", "area": "Proof foundations", "topic": "MAC precompile constraint minimisation (90→30–50)"},
        {"id": "OP-3", "area": "Proof foundations", "topic": "ZK proof of full backprop at ≥10^6 params"},
        {"id": "OP-4", "area": "Deterministic execution", "topic": "Deterministic attention backward <5% overhead"},
        {"id": "OP-5", "area": "Deterministic execution", "topic": "Multi-architecture precompile specification"},
        {"id": "OP-6", "area": "Hardware trust", "topic": "Open-hardware network TAP at line rate"},
        {"id": "OP-7", "area": "Hardware trust", "topic": "Wire-to-tensor mapping for NCCL"},
        {"id": "OP-8", "area": "Hardware trust", "topic": "Intra-node NVLink witness"},
        {"id": "OP-9", "area": "Hardware trust", "topic": "SDC vs adversarial deviation"},
        {"id": "OP-10", "area": "Protocol extensions", "topic": "Mixture-of-experts verification"},
        {"id": "OP-11", "area": "Protocol extensions", "topic": "RL post-training verification"},
        {"id": "OP-12", "area": "Protocol extensions", "topic": "Multi-site training verification"},
        {"id": "OP-13", "area": "Protocol tooling", "topic": "PyTorch/JAX → arch_spec converter"},
    ]


def prior_paradigm_overhead_estimate() -> dict[str, Any]:
    """Baker et al. ~5×10^5 training overhead vs target single-digit %."""
    return {
        "prior_zkml_training_overhead_factor": 500_000,
        "target_training_overhead_pct": [2, 10],
        "reduction_orders_of_magnitude_vs_prior": 4,
    }


def recommended_sample_k(cfg: ZkTrainingConfig | None = None) -> int:
    cfg = cfg or ZkTrainingConfig()
    return sample_count(miss_prob=cfg.sample_miss_target, deviation_fraction=cfg.sample_deviation_fraction)
