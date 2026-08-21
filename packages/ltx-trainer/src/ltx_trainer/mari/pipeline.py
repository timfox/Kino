"""MARI pipeline and CPU evaluation demo (arXiv:2605.28722)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.mari.adapters import MultiAdapterBank, competitive_train_step
from ltx_trainer.mari.baselines import (
    INFERENCE_OVERHEAD_NOTE,
    TABLE1,
    TABLE2_ABLATION,
    TABLE3_TRANSFER,
    TABLE4_LATENCY,
)
from ltx_trainer.mari.config import MARIConfig
from ltx_trainer.mari.inference import MARIPipeline
from ltx_trainer.mari.routing import usage_balance_penalty
from ltx_trainer.mari.theory import improvement_condition, routing_risk_bound


@dataclass
class MCExample:
    """Synthetic multiple-choice instance at injection hidden state."""

    example_id: str
    hidden: np.ndarray
    label: int
    applicable: bool
    category: str = "truthfulqa"


def _make_options(rng: np.random.Generator, d: int, label: int) -> np.ndarray:
    opts = rng.standard_normal((4, d))
    opts /= np.linalg.norm(opts, axis=1, keepdims=True) + 1e-9
    opts[label] *= 1.5
    return opts


def synthetic_corpus(cfg: MARIConfig, *, seed: int = 0, n: int = 24) -> list[MCExample]:
    """TruthfulQA-style MC items with heterogeneous correction needs."""
    rng = np.random.default_rng(seed)
    d = cfg.hidden_dim
    out: list[MCExample] = []
    for i in range(n):
        h = rng.standard_normal(d) * (0.5 + 0.1 * (i % 5))
        label = i % 4
        applicable = (i % 3) != 0
        out.append(
            MCExample(
                example_id=f"mc-{i}",
                hidden=h.astype(np.float64),
                label=label,
                applicable=applicable,
            )
        )
    return out


def framework_card(cfg: MARIConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MARIConfig()
    return {
        "name": "MARI",
        "title": "Multi-Adapter Representation Interventions via Energy Calibration",
        "arxiv": cfg.arxiv,
        "github": cfg.github,
        "paradigm": "representation_intervention",
        "components": [
            "competitive_multi_adapter (Eq. 5–11)",
            "energy_based_gate (Eq. 12–16)",
            "entropy_router_inference (Eq. 10)",
            "off_subspace_probe_regularizer (Eq. 14–15)",
        ],
        "config": cfg.__dict__,
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "positioning": (
            "Input-adaptive low-rank hidden-state editors with competitive K-expert "
            "training and label-free energy gating; base LLM weights stay frozen."
        ),
        "vs_reft": "ReFT uses one global adapter; MARI routes among K specialists.",
        "vs_steering": "CAA/ITI apply fixed vectors; MARI adapts direction/strength per sample.",
        "benchmarks": ["TruthfulQA MC1/MC2", "BBQ", "Sorry-Bench refusal", "MMLU", "ARC"],
        "key_hyperparameters": {
            "K": "num_adapters (typically 2–4)",
            "rho": "target_rejection_rate for energy threshold (often 0.9)",
            "r_probe": "probe_rank (paper default 2)",
        },
        "limitations_stub": (
            "This package implements numpy reference mechanics, not HuggingFace hook training "
            "on Llama/Qwen checkpoints."
        ),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "table2_ablation": TABLE2_ABLATION,
        "table3_transfer": TABLE3_TRANSFER,
        "table4_latency": TABLE4_LATENCY,
        "inference_note": INFERENCE_OVERHEAD_NOTE,
    }


def evaluation_demo(
    *,
    seed: int = 0,
    cfg: MARIConfig | None = None,
    include_train: bool = True,
) -> dict[str, Any]:
    """Run gate calibration + competitive routing on synthetic MC corpus."""
    cfg = cfg or MARIConfig(hidden_dim=64, num_adapters=3, adapter_rank=4, probe_rank=2, pca_rank=8)
    rng = np.random.default_rng(seed)
    corpus = synthetic_corpus(cfg, seed=seed, n=32)
    pipe = MARIPipeline.from_config(cfg, seed=seed)

    # Per-example option heads (toy)
    h_stack = np.stack([ex.hidden for ex in corpus])
    applicable = np.array([ex.applicable for ex in corpus], dtype=bool)
    tau = pipe.calibrate_gate(h_stack, applicable, rho=cfg.target_rejection_rate)

    train_winners: list[int] = []
    infer_adapters: list[int] = []
    correct_base = 0
    correct_mari = 0
    gated_off = 0

    for ex in corpus:
        opts = _make_options(rng, cfg.hidden_dim, ex.label)
        step = competitive_train_step(pipe.bank, ex.hidden, opts, ex.label)
        train_winners.append(step.winner)

        base_pred = int(np.argmax(opts @ ex.hidden))
        if base_pred == ex.label:
            correct_base += 1

        res = pipe.forward_mc(ex.hidden, option_embeddings=opts)
        if res.adapter_index < 0:
            gated_off += 1
        else:
            infer_adapters.append(res.adapter_index)
        if res.prediction == ex.label:
            correct_mari += 1

    usage = np.bincount(train_winners, minlength=cfg.num_adapters).astype(float)
    usage /= max(usage.sum(), 1.0)
    div = pipe.bank.diversity_loss(corpus[0].hidden)

    r_min = 0.2
    r_single = 0.35
    eta = 0.12
    theory = {
        "routing_bound": routing_risk_bound(r_min, misrouting_rate=eta),
        "improvement": improvement_condition(r_single, r_min, misrouting_rate=eta),
    }

    out: dict[str, Any] = {
        "framework": framework_card(cfg),
        "energy_threshold": tau,
        "n_examples": len(corpus),
        "accuracy_base": correct_base / len(corpus),
        "accuracy_mari": correct_mari / len(corpus),
        "fraction_gated_off": gated_off / len(corpus),
        "usage_balance_penalty": usage_balance_penalty(usage),
        "adapter_diversity": div,
        "theory": theory,
        "paper_mari_llama3_mc1": next(
            r["TruthfulQA_MC1"] for r in TABLE1 if r["method"] == "MARI" and r["model"] == "Llama-3-8B"
        ),
    }
    if include_train:
        from ltx_trainer.mari.training import demo_train

        out["competitive_train"] = demo_train(cfg=cfg, seed=seed, n=min(8, len(corpus)))
    return out
