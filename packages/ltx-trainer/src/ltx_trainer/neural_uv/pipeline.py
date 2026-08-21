"""Framework card, evaluation demo, and smoke entry points."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.neural_uv.benchmarks import benchmarks_bundle, table1_ours_rows
from ltx_trainer.neural_uv.config import NeuralUVConfig
from ltx_trainer.neural_uv.integration import integration_bundle
from ltx_trainer.neural_uv.jacobian import jacobian_demo
from ltx_trainer.neural_uv.lbo import lbo_demo
from ltx_trainer.neural_uv.ntk_lbo import ntk_lbo_demo
from ltx_trainer.neural_uv.siren import siren_demo
from ltx_trainer.neural_uv.solver import solver_demo
from ltx_trainer.neural_uv.tutte import tutte_demo


def framework_card(cfg: NeuralUVConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NeuralUVConfig()
    return {
        "paper": benchmarks_bundle()["paper"],
        "method": {
            "representation": "SIREN f_θ(x_i) → u_i with x_i = [p̃ || ψ_1:k]",
            "objective": "C2-extended Symmetric Dirichlet + injectivity barrier",
            "warmup": "Tutte embedding residual pretrain",
            "validity": "determinant-aware step rejection + fallback routing",
            "diagnostic": "NTK–LBO subspace alignment S(r)",
        },
        "config": cfg.__dict__,
        "integration": integration_bundle(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    ours = table1_ours_rows()
    return {
        "siren": siren_demo(seed=seed),
        "lbo": lbo_demo(seed=seed),
        "jacobian": jacobian_demo(seed=seed),
        "tutte": tutte_demo(seed=seed),
        "ntk_lbo": ntk_lbo_demo(seed=seed),
        "solver": solver_demo(seed=seed),
        "ref_hand_flip_pct": next(r["flip_pct"] for r in ours if r["mesh"] == "Hand"),
        "ref_thingi_valid": "42/47",
        "compact_zero_flip_count": len(ours),
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    return {
        "package": "neural_uv",
        "paper": "neural_uv",
        "arxiv": "2606.10050",
        "ref_hand_esd": 12.73,
        "ref_hand_flip_pct": 0.0,
        "solver_flip_pct": demo["solver"]["flip_pct"],
        "thingi_stratified_valid": demo["ref_thingi_valid"],
        "gopex_stub_count": len(integration_bundle()["gopex_stubs"]),
    }
