"""Framework card, demos, and training smoke."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.ssm_moe.config import SSMMoEConfig
from ltx_trainer.ssm_moe.loss import ssm_moe_total_loss
from ltx_trainer.ssm_moe.metrics import benchmarks_bundle
from ltx_trainer.ssm_moe.model import SSMMoEModel


def framework_card(cfg: SSMMoEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SSMMoEConfig()
    modes = {
        "moe_mamba": "SelectiveSSM token-mixer + sparse MoE FFN (MoE-Mamba, arXiv:2401.04081)",
        "swimba": "MoE-parameterized SSM streams, single recurrence (Swimba, arXiv:2603.06938)",
        "routing_mamba": "Shared-router projection experts + SSM core (RoM, arXiv:2506.18145)",
    }
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "mode": cfg.mode,
        "modes": modes,
        "components": [
            "SelectiveSSM diagonal stable recurrence with input-dependent Δ, B, C",
            "Top-k MoE router + Switch load-balancing aux loss",
            "Swimba parameter-space expert stream mixing (single hidden trajectory)",
            "Routing Mamba shared gate over projection expert banks",
        ],
        "related": [
            "arXiv:2510.26182 MossNet — MoE over SSM kernels as multi-head attention",
            "arXiv:2506.18145 Routing Mamba — projection MoE scaling",
            "arXiv:2603.06938 Swimba — switch Mamba MoE-parameterized SSM",
            "arXiv:2401.04081 MoE-Mamba — Mamba + MoE FFN interleaving",
        ],
        "hyperparams": {
            "d_model": cfg.d_model,
            "d_state": cfg.d_state,
            "num_experts": cfg.num_experts,
            "top_k": cfg.top_k,
            "n_layers": cfg.n_layers,
        },
    }


def paper_limitations() -> list[str]:
    return [
        "Sequential scan stub — not a fused CUDA/Triton selective scan kernel.",
        "No distributed expert parallelism or capacity dropping (Switch).",
        "Paper-scale pretraining (trillions of tokens) is external.",
        "LTX integration is architectural only; not wired into Gemma/Mamba serving stacks.",
    ]


def evaluation_demo(*, cfg: SSMMoEConfig | None = None, seq_len: int = 32, batch: int = 2) -> dict[str, Any]:
    cfg = cfg or SSMMoEConfig(d_model=64, d_state=8, d_inner=128, n_layers=2, num_experts=4, top_k=2)
    torch.manual_seed(11)
    model = SSMMoEModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    labels = ids.clone()
    logits, router_logits = model(ids)
    losses = ssm_moe_total_loss(
        logits,
        labels,
        router_logits,
        num_experts=cfg.num_experts,
        load_balance_coeff=cfg.load_balance_coeff,
    )
    params = model.num_parameters()
    return {
        "mode": cfg.mode,
        "logits_shape": list(logits.shape),
        "n_router_tensors": len(router_logits),
        "loss_total": float(losses["total"].detach()),
        "loss_lm": float(losses["lm"].detach()),
        "loss_load_balance": float(losses["load_balance"].detach()),
        "params": params,
        "benchmarks": benchmarks_bundle(),
    }


def training_step_demo(*, mode: str = "moe_mamba") -> dict[str, Any]:
    cfg = SSMMoEConfig(mode=mode, d_model=48, d_state=8, d_inner=96, n_layers=1, num_experts=4, top_k=2)
    demo = evaluation_demo(cfg=cfg, seq_len=16, batch=1)
    return {"training_step": demo}


def evaluation_smoke(cfg: SSMMoEConfig | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {"package": "ltx_trainer.ssm_moe", "ok": True, "modes": {}}
    for mode in ("moe_mamba", "swimba", "routing_mamba"):
        c = SSMMoEConfig(mode=mode, d_model=32, d_state=4, d_inner=64, n_layers=1, num_experts=4, top_k=2)
        demo = evaluation_demo(cfg=c, seq_len=8, batch=1)
        results["modes"][mode] = {
            "loss_total": demo["loss_total"],
            "logits_shape": demo["logits_shape"],
        }
    results["benchmarks"] = benchmarks_bundle()
    results["framework"] = framework_card(cfg or SSMMoEConfig())
    return results
