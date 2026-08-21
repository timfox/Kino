"""Framework card and paper benchmark excerpts (arXiv:2605.22746)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.plug_edl.config import PlugEdlConfig
from ltx_trainer.plug_edl.layout import LIMITATIONS
from ltx_trainer.plug_edl.mock import evaluation_smoke


def framework_card(cfg: PlugEdlConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PlugEdlConfig()
    return {
        "name": cfg.title,
        "paper": cfg.paper_arxiv,
        "authors": "Berk Hayta, Hannah Laus, Simon Mittermaier, Felix Krahmer",
        "framework": {
            "classical_edl": "ℓ_EDL(α,y) = E_{π~Dir(α)}[ℓ(π,y)] with digamma CE or MSE+variance",
            "plug_in_edl": "ℓ_plug(α,y) = ℓ(Π(α), y) where Π(α)=α/α0",
            "softmax_case": "τ=exp, ϕ=e ⇒ Π(α) is standard softmax (Theorem 1)",
            "uncertainty": "vacuity K/α0 and normalized predictive entropy",
        },
        "benchmark": {
            "dataset": "Google Speech Commands v1 (30-class)",
            "backbone": "MatchboxNet (NeMo)",
            "protocol": "selective prediction via uncertainty thresholding",
        },
        "headlines": headline_results(),
        "limitations": LIMITATIONS,
    }


def table1_model_variants() -> list[dict[str, Any]]:
    """Table 1: experimental model variants (abbreviated)."""
    return [
        {
            "model": "EDL-CE",
            "tau": "softplus",
            "phi": "e+1",
            "loss": "Dirichlet CE + KL",
            "kl_ramp_T": 400,
        },
        {
            "model": "Plug-in EDL-CE",
            "tau": "softplus",
            "phi": "e+1",
            "loss": "CE( p̂ )",
            "kl_ramp_T": None,
        },
        {
            "model": "Softmax",
            "tau": "exp",
            "phi": "e",
            "loss": "CE( p̂ )",
            "kl_ramp_T": None,
        },
        {
            "model": "Softmax + KL",
            "tau": "exp",
            "phi": "e",
            "loss": "CE( p̂ ) + KL",
            "kl_ramp_T": 400,
        },
        {
            "model": "EDL-MSE",
            "tau": "softplus",
            "phi": "e+1",
            "loss": "Dirichlet MSE + KL",
            "kl_ramp_T": 600,
        },
        {
            "model": "Plug-in EDL-MSE",
            "tau": "softplus",
            "phi": "e+1",
            "loss": "MSE( p̂ )",
            "kl_ramp_T": None,
        },
    ]


def table2_selective_prediction_entropy() -> list[dict[str, Any]]:
    """Table 2 excerpt: total accuracy at selective operating points (entropy)."""
    return [
        {
            "model": "Softmax",
            "base_acc": 97.21,
            "acctotal_99_0": 96.47,
            "acctotal_99_5": 94.50,
            "acctotal_99_9": 88.41,
        },
        {
            "model": "Softplus",
            "base_acc": 97.07,
            "acctotal_99_0": 96.14,
            "acctotal_99_5": 94.81,
            "acctotal_99_9": 87.64,
        },
        {
            "model": "Plug-in EDL-CE",
            "base_acc": 96.84,
            "acctotal_99_0": 95.68,
            "acctotal_99_5": 93.39,
            "acctotal_99_9": 83.55,
        },
        {
            "model": "EDL-CE",
            "base_acc": 96.88,
            "acctotal_99_0": 95.76,
            "acctotal_99_5": 93.61,
            "acctotal_99_9": 81.61,
        },
        {
            "model": "EDL-MSE",
            "base_acc": 96.55,
            "acctotal_99_0": 94.91,
            "acctotal_99_5": 92.87,
            "acctotal_99_9": 80.93,
        },
    ]


def table2_selective_prediction_vacuity_excerpt() -> list[dict[str, Any]]:
    """Table 2 excerpt: vacuity-based Acctotal at 99.9% Accth target."""
    return [
        {"model": "Softmax", "acctotal_99_9": 62.00},
        {"model": "Softmax + KL", "acctotal_99_9": 80.36},
        {"model": "EDL-CE", "acctotal_99_9": 81.62},
        {"model": "Plug-in EDL-CE", "acctotal_99_9": 83.55},
        {"model": "EDL-CE no KL", "acctotal_99_9": 47.14},
    ]


def headline_results() -> dict[str, Any]:
    cfg = PlugEdlConfig()
    return {
        "softmax_base_acc_pct": cfg.softmax_base_acc_pct,
        "softmax_entropy_acctotal_at_99_9_accth_pct": cfg.softmax_entropy_acctotal_99_9_pct,
        "plug_in_tracks_classical_edl": True,
        "kl_improves_vacuity_selective_prediction": True,
        "first_edl_coverage_accuracy_speech": True,
    }


def evaluation_demo(cfg: PlugEdlConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PlugEdlConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_model_variants": table1_model_variants(),
        "table2_entropy": table2_selective_prediction_entropy(),
        "table2_vacuity_excerpt": table2_selective_prediction_vacuity_excerpt(),
        "headlines": headline_results(),
    }
