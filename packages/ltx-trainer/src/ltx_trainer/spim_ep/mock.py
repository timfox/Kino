"""Runnable evaluation smoke for SPIM-EP (arXiv:2606.13454)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.spim_ep.benchmarks import TABLE_WINE_EXPERIMENT, benchmarks_bundle
from ltx_trainer.spim_ep.config import FINITE_DIFF_DELTA, PAPER_ARXIV, SPIMEPConfig
from ltx_trainer.spim_ep.energy import spim_hamiltonian
from ltx_trainer.spim_ep.mattis import coupling_from_mattis, init_mattis, tilde_coupling
from ltx_trainer.spim_ep.nonlinear import rho
from ltx_trainer.spim_ep.pipeline import evaluation_demo_run


def evaluation_smoke() -> dict[str, Any]:
    cfg = SPIMEPConfig(n_free=3, n_nudge=2, rank=4)
    params, j = init_mattis(cfg, seed=0)
    s = np.zeros(cfg.n_dynamic)
    j_dyn = j[cfg.n_input :, cfg.n_input :]
    h0 = spim_hamiltonian(s, j_dyn)
    j_t = tilde_coupling(j_dyn)
    h_tilde = spim_hamiltonian(s, j_t)
    demo = evaluation_demo_run(cfg, seed=1)
    return {
        "package": "spim_ep",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_wine_test_acc_pct": TABLE_WINE_EXPERIMENT["test_accuracy_pct"],
        "ref_mnist_acc_pct": benchmarks_bundle()["mnist_all_to_all"]["test_accuracy_pct"],
        "finite_diff_delta": FINITE_DIFF_DELTA,
        "hamiltonian_at_origin": h0,
        "hamiltonian_tilde_at_origin": h_tilde,
        "rho_at_zero": float(rho(np.array([0.0]))[0]),
        "n_spim_per_step": cfg.n_spim_evaluations(),
        "demo_test_accuracy": demo["test_accuracy"],
        "demo_data_source": demo["data_source"],
    }
