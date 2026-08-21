"""MAS-IB evaluation pipeline and smoke checks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mas_ib.config import MasIbConfig
from ltx_trainer.mas_ib.ib import evaluate_relay, infinite_bandwidth_equivalent, mas_gain
from ltx_trainer.mas_ib.prototypes import compare_prototypes


def run_regime_demo() -> dict[str, Any]:
    """Low-δ (near-sufficient) vs high-δ (lossy) relays across β."""
    regimes = {}
    # δ≈0: small loss, compression helps weak models
    for name, beta, delta, relay_bits in (
        ("low_delta_weak", 0.5, 0.02, 10.0),
        ("low_delta_strong", 3.0, 0.02, 10.0),
        ("high_delta_weak", 0.5, 30.0, 40.0),
        ("high_delta_strong", 3.0, 30.0, 40.0),
    ):
        cfg = MasIbConfig(beta=beta, delta_loss=delta, relay_bits=relay_bits, full_context_bits=64.0)
        stats = evaluate_relay(cfg)
        cmp_ = compare_prototypes(config=cfg)
        regimes[name] = {
            "relay": stats.as_dict(),
            "mas_helps_theory": stats.mas_gain > 0,
            "mas_minus_contextflow": cmp_["mas_minus_contextflow"],
        }
    return regimes


def evaluation_demo() -> dict[str, Any]:
    weak_ok = compare_prototypes(config=MasIbConfig(beta=0.5, delta_loss=0.05))
    # High β + large Δ so H(M|m) < βΔ (Theorem 4.1 predicts MAS hurts)
    strong_cfg = MasIbConfig(beta=4.0, delta_loss=20.0, relay_bits=48.0, full_context_bits=64.0)
    strong_lossy = compare_prototypes(config=strong_cfg)
    equiv = infinite_bandwidth_equivalent()
    regimes = run_regime_demo()
    return {
        "weak_near_sufficient": weak_ok,
        "strong_lossy": strong_lossy,
        "infinite_bandwidth": equiv,
        "regimes": regimes,
        "weak_helps": weak_ok["mas_helps"],
        "strong_lossy_hurts_or_shrinks": strong_lossy["mas_minus_contextflow"]
        <= weak_ok["mas_minus_contextflow"],
        "theorem_holds_weak": evaluate_relay(MasIbConfig(beta=0.5, delta_loss=0.05)).mas_gain > 0,
        "theorem_holds_strong_lossy": evaluate_relay(strong_cfg).mas_gain < 0,
    }


def evaluation_smoke() -> dict[str, bool]:
    from ltx_trainer.mas_ib.benchmarks import PAPER_ANCHORS, TABLE_2_GAINS

    demo = evaluation_demo()
    g_pos = mas_gain(h_given_m=56.0, beta=0.5, delta=0.05)
    g_neg = mas_gain(h_given_m=10.0, beta=4.0, delta=5.0)
    alf_qwen = next(r for r in TABLE_2_GAINS if r["task"] == "ALFWorld")
    checks = {
        "prop_31_equivalence": demo["infinite_bandwidth"]["y_mas_equals_y_sas"],
        "weak_helps": demo["weak_helps"],
        "theorem_pos": g_pos > 0,
        "theorem_neg": g_neg < 0,
        "strong_lossy_worse_gain": demo["strong_lossy_hurts_or_shrinks"],
        "theorem_holds_strong_lossy": demo["theorem_holds_strong_lossy"],
        "paper_alfworld_7b": abs(float(alf_qwen["qwen25_7b"]) - 0.194) < 1e-6,
        "paper_workbench_4o": abs(float(PAPER_ANCHORS["workbench_gpt4o_gain"]) - (-0.086)) < 1e-6,
        "paper_n_experiments": int(PAPER_ANCHORS["n_controlled_experiments"]) == 18,
        "paper_n_benchmarks": int(PAPER_ANCHORS["n_benchmarks"]) == 5,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def evaluation_smoke_json() -> dict[str, bool]:
    return evaluation_smoke()
