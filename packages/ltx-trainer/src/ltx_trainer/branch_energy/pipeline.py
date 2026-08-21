"""Demo pipeline aggregating Section-IV cases."""

from __future__ import annotations

from typing import Any

from ltx_trainer.branch_energy.benchmarks import summary_anchors
from ltx_trainer.branch_energy.config import BranchEnergyConfig
from ltx_trainer.branch_energy.test_cases import run_all_cases


def run_demo(cfg: BranchEnergyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BranchEnergyConfig(num_periods=1.0, fs_hz=10_000.0)
    cases = run_all_cases(cfg)
    residuals = [cases[k]["balance_residual"] for k in ("IV-A", "IV-B", "IV-C", "IV-D", "IV-F") if "balance_residual" in cases[k]]
    return {
        "config": {"f0_hz": cfg.f0_hz, "fs_hz": cfg.fs_hz, "periods": cfg.num_periods},
        "cases": cases,
        "max_balance_residual": max(residuals) if residuals else 0.0,
        "open_phase_ic_zero": cases["IV-B"]["ic_max_abs"] < 1e-12,
        "open_phase_cpc_ghost": cases["IV-B"].get("cpc_ghost_in_open_phase", False),
        "triac_no_storage": cases["IV-C"]["storage_rms"] == 0.0,
        "triac_pseudo_reactive": cases["IV-C"].get("pseudo_reactive_nontrivial", False),
        "duality_ok": cases["IV-F"]["duality_max_error_W"] < cases["IV-F"]["anchor_tol_W"],
        "topology_profiles_differ": cases["IV-D"]["profiles_differ"],
        "summary": summary_anchors(),
    }
