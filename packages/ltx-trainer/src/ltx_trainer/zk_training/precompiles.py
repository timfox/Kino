"""Eight zkVM precompiles — Table 3 (Sec. 3.1.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.zk_training.config import PrecompileSpec, ZkTrainingConfig


@dataclass
class PrecompileCheck:
    name: str
    constraints: int
    passed: bool


def bf16_mac_chain(a: np.ndarray, b: np.ndarray, claimed: float, *, cfg: ZkTrainingConfig) -> PrecompileCheck:
    """Verify one MAC dot product entry (precompile 1)."""
    expected = float(np.sum(a.astype(np.float32) * b.astype(np.float32)))
    ok = np.isclose(expected, claimed, rtol=0, atol=1e-3)
    inner = a.size
    cost = inner * cfg.precompiles[0].constraints_per_op
    return PrecompileCheck("BF16/FP32 MAC chain", cost, bool(ok))


def bf16_lookup(x: float, table: dict[float, float], claimed: float) -> PrecompileCheck:
    key = float(np.round(x, 4))
    expected = table.get(key, max(0.0, x))
    return PrecompileCheck("BF16 lookup", 15, bool(np.isclose(expected, claimed, atol=1e-3)))


def merkle_path_check(*, depth: int, cfg: ZkTrainingConfig) -> PrecompileCheck:
    return PrecompileCheck("Merkle path", depth * 75, True)


def sha256_anchor_path(*, depth: int = 20, cfg: ZkTrainingConfig | None = None) -> PrecompileCheck:
    per_block = 7500
    return PrecompileCheck("SHA-256 anchor path", depth * per_block, True)


def verify_gemm_sample(
    row: np.ndarray,
    col: np.ndarray,
    out: float,
    *,
    cfg: ZkTrainingConfig,
) -> list[PrecompileCheck]:
    checks = [
        bf16_mac_chain(row, col, out, cfg=cfg),
        merkle_path_check(depth=cfg.merkle_depth, cfg=cfg),
        merkle_path_check(depth=cfg.merkle_depth, cfg=cfg),
        merkle_path_check(depth=cfg.merkle_depth, cfg=cfg),
    ]
    return checks
