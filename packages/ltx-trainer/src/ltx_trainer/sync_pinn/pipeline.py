"""Framework card and benchmark bundles."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sync_pinn.config import SyncPinnConfig
from ltx_trainer.sync_pinn.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.sync_pinn.mock import evaluation_smoke
from ltx_trainer.sync_pinn.tables import (
    fig2_baseline_setting,
    fig4_intrinsic_cost_trends,
    fig5_target_cost_trends,
    fig6_baseline_comparison_kuramoto,
    fig7_sakaguchi_frustrated,
    fig8_noise_robustness,
    headline_results,
)


def framework_card(cfg: SyncPinnConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SyncPinnConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "author": cfg.author,
        "problem": (
            "Regulate when synchronization emerges (t*) and coherence level (R*) "
            "in continuous-time networked oscillators without prescribing a feedback law."
        ),
        "parameterization": {
            "state_control": "x(t)=x(0)+h(t)Nx(t), u(t)=u(0)+h(t)Nu(t) (Eq. 2)",
            "residual": "r(t)=dx/dt−f(p,x,u) via autodiff in full PINN (Eq. 3)",
            "loss": "L=Ldyn+Lic+Lcontrol+Lreg (Eq. 4)",
        },
        "kuramoto": {
            "dynamics": "θ̇_i = ω_i + K Σ_j A_ij sin(θ_j−θ_i) + u_i (Eq. 5)",
            "order_parameter": "R(t)=|(1/N)Σ e^{iθ_j}| (Eq. 6)",
            "objective": "R(t)≥R* for all t≥t* (Eqs. 7–8, 10)",
        },
        "baselines": [
            "Linear phase feedback u_i=−k_θ(θ_i−θ̄) (Eq. 15)",
            "Nonlinear phase feedback u_i=−k_θ sin(θ_i−θ̄) (Eq. 16)",
            "Frequency compensation u_i=−k_ω(ω_i−ω̄) (Eq. 17)",
            "Kuramoto–Sakaguchi with phase lag α (Eq. 18)",
        ],
        "default_experiment": fig2_baseline_setting(),
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "fig2_baseline": fig2_baseline_setting(),
        "fig4_intrinsic_E_trends": fig4_intrinsic_cost_trends(),
        "fig5_target_E_trends": fig5_target_cost_trends(),
        "fig6_kuramoto_baselines": fig6_baseline_comparison_kuramoto(),
        "fig7_sakaguchi": fig7_sakaguchi_frustrated(),
        "fig8_noise": fig8_noise_robustness(),
        "headlines": headline_results(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"smoke": evaluation_smoke(), "headlines": headline_results()}
