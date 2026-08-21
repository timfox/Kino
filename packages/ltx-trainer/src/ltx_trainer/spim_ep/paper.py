"""Agent-facing framework card for SPIM-EP."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spim_ep.benchmarks import benchmarks_bundle
from ltx_trainer.spim_ep.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    return {
        "paper": {
            "title": PAPER_TITLE,
            "arxiv": f"arXiv:{PAPER_ARXIV}",
            "url": PAPER_URL,
            "authors": "Vanden Abeele, Veraldi, Pierangeli, Conti, Massar (ULB / Sapienza)",
        },
        "problem": (
            "Backpropagation is hard to map onto analog photonic hardware. Equilibrium Propagation (EP) "
            "trains energy-based networks via contrastive nudging without explicit backprop."
        ),
        "method": {
            "platform": "Spatial Photonic Ising Machine (SPIM) + digital remainder",
            "energy": "E = B + I + α‖s‖²/2 + β‖s_out − y‖²/2 with ρ(s)=sin(s) (saturated outside ±π/2)",
            "coupling": "Rank-K Mattis decomposition J_ij = (1/K) Σ_k λ_k ξ_{k,i} ξ_{k,j}",
            "gradients": "∂I/∂s_m via ±π/4 optical finite differences; ∂I/∂λ_k optical; ∂I/∂ξ digital/BOP",
            "learning": "Contrastive EP update ∆θ ∝ (∂E/∂θ|−β − ∂E/∂θ|+β) / (2β)",
        },
        "results": {
            "wine_test_acc_pct": benchmarks_bundle()["wine_experiment"]["test_accuracy_pct"],
            "wine_sim_acc_pct": benchmarks_bundle()["wine_experiment"]["simulation_accuracy_pct"],
            "mnist_continuous_acc_pct": benchmarks_bundle()["mnist_all_to_all"]["test_accuracy_pct"],
            "mnist_layered_acc_pct": benchmarks_bundle()["mnist_layered_architecture"]["test_accuracy_pct"],
        },
        "gopex": {
            "package": "ltx_trainer.spim_ep",
            "cli": "./scripts/kino-spim-ep.sh knowledge",
            "tools": "spim_ep_framework_card, spim_ep_eval_demo",
        },
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.spim_ep.mock import evaluation_smoke

    return evaluation_smoke()
