"""Framework card, demos, smoke for PGN NequIP (arXiv:2605.30822)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pgn_nequip.config import PGNNequIPConfig
from ltx_trainer.pgn_nequip.nequip import learning_curve_exponents, nl_lmax_sweep, parity_samples
from ltx_trainer.pgn_nequip.paper_tables import (
    cost_speedup,
    fig4_parity_mae,
    fig5_hyperuniformity_s0,
    fig5d_nearest_neighbor_experiment,
    knowledge_card,
)
from ltx_trainer.pgn_nequip.simulation import full_pipeline_demo


def framework_card(cfg: PGNNequIPConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PGNNequIPConfig()
    d = cfg.design
    n = cfg.nequip
    return {
        "name": "PGN-NequIP",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "title": "GNN many-body interactions in amorphous polymer-grafted nanoparticles",
        "task": "Classical DFT PEL → NequIP surrogate → GNN-driven MC equilibration",
        "pipeline": [
            "Hard-sphere configuration sampling (LAMMPS-style stub)",
            "Classical DFT many-body energy (Eq. 1–4)",
            "NequIP E(3)-equivariant message passing (lmax, nl, rcut)",
            "Metropolis MC with GNN energies",
            "Structure: g(r), S(q), Steinhardt BOO",
        ],
        "design": {
            "n_particles": d.n_particles,
            "phi_c": d.phi_c,
            "sigma_g_chains_nm2": d.sigma_g_chains_nm2,
            "mw_kda": d.mw_kda,
        },
        "nequip": {
            "lmax": n.lmax,
            "n_layers": n.n_layers,
            "rcut_over_d": n.rcut_over_d,
            "frames_train": n.train_frames,
        },
        "cost_speedup": cost_speedup(),
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy stub — no LAMMPS, Lebedev quadrature, or mir-group/nequip training.",
        "Classical DFT is a free-energy proxy, not the full Eq. (1) volume integrals.",
        "NequIP predictions are parity-style noise models anchored to Fig. 4 MAE/σ_E.",
        "MC uses short equilibration; experimental SAXS curves are table anchors only.",
        "Steinhardt BOO is a simplified proxy, not Freud library outputs.",
    ]


def evaluation_demo(cfg: PGNNequIPConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PGNNequIPConfig()
    demo = full_pipeline_demo(cfg=cfg)
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "knowledge": knowledge_card(),
        "fig4_parity": fig4_parity_mae(),
        "fig5_s0_limit": fig5_hyperuniformity_s0(),
        "fig5d_rnn": fig5d_nearest_neighbor_experiment(),
        "learning_exponents": learning_curve_exponents(),
        "nl_lmax_sweep": nl_lmax_sweep(),
        "parity_samples": parity_samples(cfg),
        "pipeline_demo": demo,
    }


def evaluation_smoke(cfg: PGNNequIPConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    cfg = cfg or PGNNequIPConfig()
    pipe = demo["pipeline_demo"]
    rcut = pipe["rcut_sweep"]
    exponents = demo["learning_exponents"]

    assert pipe["mae_over_sigma"] < 0.05
    assert rcut[-2]["mae_over_sigma"] < 0.02
    assert rcut[0]["mae_over_sigma"] > rcut[-1]["mae_over_sigma"]
    assert exponents[3] > exponents[0]
    assert pipe["speedup_vs_dft"] >= 1e4
    assert abs(pipe["sq_at_low_q"] - 0.5) < 0.15
    assert cfg.nequip.lmax == 3 and cfg.nequip.n_layers == 3

    best = min(demo["fig4_parity"], key=lambda r: r["mae_over_sigma"])
    assert best["mae_over_sigma"] <= 0.011

    return {
        "status": "ok",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "mae_over_sigma": pipe["mae_over_sigma"],
        "speedup_vs_dft": pipe["speedup_vs_dft"],
        "learning_exponent_lmax3": exponents[3],
        "demo_keys": list(demo.keys()),
    }
