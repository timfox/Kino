"""ZK frontier training verification config (arXiv:2606.05433)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegulatoryThresholds:
    eu_ai_act_flops: float = 1e25
    us_eo_14110_flops: float = 1e26
    llama31_405b_flops: float = 3.8e25
    training_budget_usd: float = 100e6


@dataclass(frozen=True)
class PrecompileSpec:
    index: int
    name: str
    constraints_per_op: int
    role: str


DEFAULT_PRECOMPILES: tuple[PrecompileSpec, ...] = (
    PrecompileSpec(1, "BF16/FP32 MAC chain", 90, "GEMM dot products"),
    PrecompileSpec(2, "BF16 lookup table", 15, "GELU, SiLU, derivatives"),
    PrecompileSpec(3, "Merkle path (Poseidon)", 1500, "Bind hints to commitments"),
    PrecompileSpec(4, "FP32 exp", 250, "Softmax, cross-entropy"),
    PrecompileSpec(5, "FP32 sqrt/rsqrt", 200, "RMSNorm, Adam"),
    PrecompileSpec(6, "FP32 div", 200, "Normalisation, Adam"),
    PrecompileSpec(7, "FP32 log", 250, "Cross-entropy loss"),
    PrecompileSpec(8, "SHA-256 compression", 7500, "Network-anchor reconciliation"),
)


@dataclass
class ZkTrainingConfig:
    paper_arxiv: str = "arXiv:2606.05433"
    packages: tuple[str, ...] = ("RISC Zero zkVM", "Poseidon Merkle", "network TAP / SmartNIC")
    thresholds: RegulatoryThresholds = field(default_factory=RegulatoryThresholds)
    precompiles: tuple[PrecompileSpec, ...] = DEFAULT_PRECOMPILES
    sample_k_default: int = 4605
    sample_miss_target: float = 1e-20
    sample_deviation_fraction: float = 0.01
    merkle_depth: int = 30
    anchor_leaf_bytes: int = 16384
    determinism_overhead_pct: tuple[float, float] = (1.6, 8.2)
    merkle_hash_overhead_pct: tuple[float, float] = (0.5, 1.5)
    aggregate_proof_kb: int = 200
    deployment_months_poc: int = 36
    open_problems: tuple[str, ...] = tuple(f"OP-{i}" for i in range(1, 14))
    demo_layers: int = 2
    demo_hidden: int = 1024
    demo_samples_per_tensor: int = 100
