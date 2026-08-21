"""Framework card and benchmarks for sparse 4D bootstrapped cross-validation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sparse4d_bcv.config import Sparse4dBcvConfig
from ltx_trainer.sparse4d_bcv.layout import (
    COMPARISON_MODES,
    EVALUATION_METRICS,
    LIMITATIONS,
    RECOMMENDED_PRACTICE,
)
from ltx_trainer.sparse4d_bcv.mock import bootstrap_demo, evaluation_smoke, nyquist_demo
from ltx_trainer.sparse4d_bcv.tables import (
    compatible_reconstructors,
    sparse_regime_findings,
    table1_metrics_summary,
    table_s1_ultrasparse_angles,
    ultrasparse_regime_findings,
)


def framework_card(cfg: Sparse4dBcvConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Sparse4dBcvConfig()
    return {
        "name": "Bootstrapped cross-validation for sparse 4D reconstruction",
        "paper": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "institutions": list(cfg.institutions),
        "problem": (
            "Reference-free assessment of sparse / ultra-sparse 4D (3D+time) reconstructions "
            "when ground-truth y is unavailable, inspired by cryo-EM split-map validation."
        ),
        "method": {
            "pseudo_reference": "Full-set reconstruction ỹ from all available projections/experiments",
            "bootstrap": f"{cfg.num_bootstrap_subsets} independent subsets; "
            f"{cfg.num_cv_pairs} cross-validation pairs",
            "cross_validation": "C = M(ŷ_a, ŷ_b) with interlaced time for 4D independence (Eq. 5)",
            "nyquist_velocity": "v_Nyquist = Δx / Δt (temporal Nyquist assumed satisfied)",
        },
        "evaluation_metrics": list(EVALUATION_METRICS),
        "comparison_modes": list(COMPARISON_MODES),
        "regimes": {
            "sparse": {
                "description": "~10–20% Crowther projections (scanning tomography)",
                "bootstrap_axis": "projection count",
                "counts": list(cfg.sparse_projection_counts),
                "full_projections": cfg.num_projections_full,
            },
            "ultrasparse": {
                "description": "<5% views (scanning-free multi-projection)",
                "bootstrap_axis": "experiment count",
                "counts": list(cfg.ultrasparse_experiment_counts),
                "projections_per_experiment": cfg.ultrasparse_projections_per_experiment,
            },
        },
        "demo_dataset": {
            "name": cfg.dataset_name,
            "doi": cfg.dataset_doi,
            "experiments": cfg.num_experiments,
            "frames": cfg.num_time_steps,
            "shape": list(cfg.volume_shape),
            "simulation": "Navier–Stokes–Cahn–Hilliard (DUNE-FEM)",
            "example_reconstructor": cfg.example_reconstructor,
        },
        "compatible_reconstructors": compatible_reconstructors(),
        "recommended_practice": list(RECOMMENDED_PRACTICE),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_metrics": table1_metrics_summary(),
        "table_s1_angles": table_s1_ultrasparse_angles(),
        "sparse_regime_findings": sparse_regime_findings(),
        "ultrasparse_regime_findings": ultrasparse_regime_findings(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {
        "smoke": evaluation_smoke(),
        "nyquist": nyquist_demo(),
        "bootstrap": bootstrap_demo(),
    }
