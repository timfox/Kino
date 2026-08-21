"""Neighbor-consistent PSZ stub (arXiv:2605.21891)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NcPszConfig:
    paper_arxiv: str = "arXiv:2605.21891"
    title: str = "Neighbor-consistent neural filters for robust PSZ under localization uncertainty"
    num_listeners: int = 2
    coord_dim: int = 2
    nc_delta_m: float = 0.01
    nc_lambda_woofer: float = 0.75
    nc_lambda_tweeter: float = 0.75
    overlap_threshold_m: float = 0.5  # dov placeholder for regime mask
    simulation_anchors: int = 25
    sim_grid_rmax_m: float = 0.10
    sim_grid_step_m: float = 0.01
    # Headline simulation improvements (woofer band, Listener 2)
    woofer_ipi_sigma_rms_imp_pct: float = 55.9
    tweeter_izi_sigma_rms_imp_pct: float = 30.3
    meas_l2_izi_min_imp_pct: float = 16.9
    meas_l1_izi_sigma_mean_imp_pct: float = 61.8
