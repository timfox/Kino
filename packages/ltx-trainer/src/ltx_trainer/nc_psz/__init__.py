"""Neighbor-consistent PSZ stub (arXiv:2605.21891)."""

from ltx_trainer.nc_psz.config import NcPszConfig
from ltx_trainer.nc_psz.layout import LIMITATIONS
from ltx_trainer.nc_psz.losses import (
    baseline_psz_loss_toy,
    clip_coordinates,
    neighbor_consistency_loss,
    same_region_mask,
    total_training_loss,
)
from ltx_trainer.nc_psz.metrics import (
    improvement_quality_pct,
    improvement_stability_pct,
    isolation_ratio_db,
    neighborhood_cvar10,
    neighborhood_median,
    neighborhood_min,
    stability_variation_rates,
)
from ltx_trainer.nc_psz.mock import evaluation_smoke
from ltx_trainer.nc_psz.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table_i_simulation_woofer,
    table_ii_simulation_tweeter,
    table_iii_measurements_excerpt,
)

__all__ = [
    "LIMITATIONS",
    "NcPszConfig",
    "baseline_psz_loss_toy",
    "benchmarks_bundle",
    "clip_coordinates",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "improvement_quality_pct",
    "improvement_stability_pct",
    "isolation_ratio_db",
    "neighbor_consistency_loss",
    "neighborhood_cvar10",
    "neighborhood_median",
    "neighborhood_min",
    "same_region_mask",
    "stability_variation_rates",
    "table_i_simulation_woofer",
    "table_ii_simulation_tweeter",
    "table_iii_measurements_excerpt",
    "total_training_loss",
]
