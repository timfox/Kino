"""Genesis, in-training step, and ex-ante attestation (P.1–P.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ltx_trainer.zk_training.commitments import AnchorChain, PreCommitment, merkle_root
from ltx_trainer.zk_training.config import ZkTrainingConfig
from ltx_trainer.zk_training.precompiles import bf16_lookup, verify_gemm_sample


@dataclass
class ExAnteClaims:
    max_total_flops: float
    max_steps: int
    training_regime: str = "dense_pretrain"
    data_filter_hash: bytes = b"\x00" * 32

    def hash(self) -> bytes:
        import hashlib

        h = hashlib.sha256()
        h.update(str(self.max_total_flops).encode())
        h.update(str(self.max_steps).encode())
        h.update(self.training_regime.encode())
        h.update(self.data_filter_hash)
        return h.digest()


@dataclass
class StepRecord:
    step: int
    cumulative_flops: float
    weight_root: bytes
    checks: list[str] = field(default_factory=list)


@dataclass
class ProtocolRun:
    h_commit: bytes
    genesis_pass: bool
    step_records: list[StepRecord]
    ex_ante_ok: bool
    anchor_terminal: bytes
    proof_constraints: int


def _relu_table() -> dict[float, float]:
    return {float(x): float(max(0.0, x)) for x in np.linspace(-4, 4, 33)}


def run_toy_mlp_protocol(*, cfg: ZkTrainingConfig, steps: int = 3, seed: int = 0) -> ProtocolRun:
    """Appendix E two-layer MLP with pipeline-parallel messaging."""
    rng = np.random.default_rng(seed)
    din = cfg.demo_hidden
    dh = cfg.demo_hidden
    dout = cfg.demo_hidden

    w0 = merkle_root(rng.standard_normal((din + dh + dout, din)))
    data_root = merkle_root(rng.standard_normal((8, din)))
    arch_hash = merkle_root(np.array([din, dh, dout], dtype=np.int64))
    claims = ExAnteClaims(max_total_flops=cfg.thresholds.llama31_405b_flops, max_steps=steps + 10)
    pre = PreCommitment(arch_hash, data_root, w0, claims.hash())
    chain = AnchorChain(anchor_init=merkle_root(np.array([1.0])))

    x = rng.standard_normal((4, din)).astype(np.float32)
    w1 = rng.standard_normal((dh, din)).astype(np.float32)
    w2 = rng.standard_normal((dout, dh)).astype(np.float32)
    relu = _relu_table()

    records: list[StepRecord] = []
    constraints = 0
    flops_per_step = float(din * dh + dh * dout) * 4.0
    cumulative = 0.0

    for t in range(1, steps + 1):
        mlp1_out = x @ w1.T
        act1 = np.vectorize(lambda v: relu.get(float(np.round(v, 4)), max(0.0, v)))(mlp1_out)
        mlp2_out = act1 @ w2.T
        act2 = np.vectorize(lambda v: relu.get(float(np.round(v, 4)), max(0.0, v)))(mlp2_out)
        msg_fwd = merkle_root(act1)
        msg_bwd = merkle_root(mlp2_out - rng.standard_normal(act2.shape))

        r, j = 0, 0
        checks = verify_gemm_sample(x[r], w1[j, :], float(mlp1_out[r, j]), cfg=cfg)
        checks.append(bf16_lookup(float(mlp1_out[r, j]), relu, float(act1[r, j])))
        checks.append(bf16_lookup(float(mlp2_out[r, j]), relu, float(act2[r, j])))
        constraints += sum(c.constraints for c in checks)

        wt_root = merkle_root(w1) + merkle_root(w2)
        chain.link(t, wt_root, msg_fwd + msg_bwd)
        cumulative += flops_per_step
        records.append(
            StepRecord(
                step=t,
                cumulative_flops=cumulative,
                weight_root=wt_root,
                checks=[c.name for c in checks if c.passed],
            )
        )

    genesis_checks = verify_gemm_sample(x[0], w1[0, :], float((x @ w1.T)[0, 0]), cfg=cfg)
    genesis_pass = all(c.passed for c in genesis_checks)
    ex_ante_ok = cumulative <= claims.max_total_flops and steps <= claims.max_steps

    return ProtocolRun(
        h_commit=pre.h_commit(),
        genesis_pass=genesis_pass,
        step_records=records,
        ex_ante_ok=ex_ante_ok,
        anchor_terminal=chain.terminal(),
        proof_constraints=constraints,
    )
