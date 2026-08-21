"""VTM partition survey framework card, tables, and smoke demos (arXiv:2605.21526)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.vtm_partition.config import VTMPartitionConfig
from ltx_trainer.vtm_partition.features import extract_state_vector, feature_groups
from ltx_trainer.vtm_partition.layout import LIMITATIONS
from ltx_trainer.vtm_partition.metrics import complexity_ratio, encoding_time_ratio
from ltx_trainer.vtm_partition.mock import sample_cu_context, sample_ground_truth_costs
from ltx_trainer.vtm_partition.qtmtt import hevc_vs_vvc_scale_facts
from ltx_trainer.vtm_partition.rl_agent import (
    composite_loss,
    predict_q_values_linear,
    rl_tradeoff_point,
    select_splits_top_n,
    table_iii_rl_tradeoffs,
)


def framework_card(cfg: VTMPartitionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VTMPartitionConfig()
    return {
        "name": "VTM Partition Acceleration Survey",
        "paper": cfg.paper_arxiv,
        "idea": (
            "Survey of VVC QTMTT partitioning acceleration vs evolving VTM reference software; "
            "includes toy size-independent DQN agent that prunes split-mode RDO search."
        ),
        "reference_encoder": cfg.reference_vtm,
        "qtmtt_splits": list(cfg.split_modes),
        "rl_agent": {
            "feature_dim": cfg.rl_feature_dim,
            "top_n_inference": list(cfg.rl_top_n_choices),
            "loss": "α1·MSE1 + α2·MSE2 + α3·MSE3 (Eq. 3)",
            "training_data": "BVI-DVC + VTM-18.0 AI, 12M+ 32×32 trajectories (paper)",
        },
        "scale_facts": hevc_vs_vvc_scale_facts(),
        "defaults": cfg.__dict__,
    }


def table_i_vtm_evolution() -> list[dict[str, float | str]]:
    """Table 1 — VTM versions vs VTM-18.0 (AI), relative CU/pixel/ET and BD-rate."""
    return [
        {
            "version": "VTM-3.0",
            "cu_ratio_pct": 77.34,
            "pixel_ratio_pct": 69.75,
            "et_pct": 75.80,
            "bd_rate_y_pct": 8.60,
            "bd_rate_yuv_pct": 6.68,
        },
        {
            "version": "VTM-5.0",
            "cu_ratio_pct": 66.03,
            "pixel_ratio_pct": 76.45,
            "et_pct": 65.50,
            "bd_rate_y_pct": 4.14,
            "bd_rate_yuv_pct": 2.98,
        },
        {
            "version": "VTM-6.0",
            "cu_ratio_pct": 116.02,
            "pixel_ratio_pct": 112.66,
            "et_pct": 102.20,
            "bd_rate_y_pct": 1.72,
            "bd_rate_yuv_pct": 1.90,
        },
        {
            "version": "VTM-7.0",
            "cu_ratio_pct": 112.88,
            "pixel_ratio_pct": 109.21,
            "et_pct": 102.70,
            "bd_rate_y_pct": 1.57,
            "bd_rate_yuv_pct": 1.77,
        },
        {
            "version": "VTM-8.0",
            "cu_ratio_pct": 107.32,
            "pixel_ratio_pct": 103.92,
            "et_pct": 105.00,
            "bd_rate_y_pct": 1.97,
            "bd_rate_yuv_pct": 0.19,
        },
        {
            "version": "VTM-9.0",
            "cu_ratio_pct": 106.66,
            "pixel_ratio_pct": 103.58,
            "et_pct": 103.20,
            "bd_rate_y_pct": 0.67,
            "bd_rate_yuv_pct": 0.68,
        },
        {
            "version": "VTM-10.2",
            "cu_ratio_pct": 106.41,
            "pixel_ratio_pct": 103.39,
            "et_pct": 103.10,
            "bd_rate_y_pct": 0.65,
            "bd_rate_yuv_pct": 0.64,
        },
        {
            "version": "VTM-23.11",
            "cu_ratio_pct": 99.87,
            "pixel_ratio_pct": 99.86,
            "et_pct": 93.50,
            "bd_rate_y_pct": -0.05,
            "bd_rate_yuv_pct": -0.07,
        },
    ]


def table_ii_rl_features() -> list[dict[str, Any]]:
    """Table 2 — RL state feature groups."""
    return feature_groups()


def sota_partition_methods() -> list[dict[str, Any]]:
    """Sec. 3 — SOTA acceleration excerpts (Fig. 2 narrative)."""
    return [
        {
            "authors": "Cui et al.",
            "vtm_baseline": "VTM-5.0",
            "method": "Gradient-based early termination",
            "et_reduction_pct": 51.0,
            "bd_rate_pct": 1.2,
        },
        {
            "authors": "Tissier et al.",
            "vtm_baseline": "VTM-6.1",
            "method": "CNN boundary probabilities",
            "et_reduction_pct": 51.5,
            "bd_rate_pct": 1.45,
        },
        {
            "authors": "Li et al.",
            "vtm_baseline": "VTM-7.0",
            "method": "Pruned CNN + convex model selection",
            "et_reduction_pct": 50.0,
            "bd_rate_pct": 1.47,
        },
        {
            "authors": "Li et al. (DeepQTMT)",
            "vtm_baseline": "VTM-7.0",
            "method": "MSE-CNN + early exit",
            "et_reduction_pct": 44.65,
            "bd_rate_pct": 1.32,
        },
        {
            "authors": "Saldanha et al.",
            "vtm_baseline": "VTM-10.0",
            "method": "Light GBM binary split tasks",
            "et_reduction_pct": 61.34,
            "bd_rate_pct": 2.43,
        },
        {
            "authors": "Tissier et al.",
            "vtm_baseline": "VTM-10.2",
            "method": "CNN + decision tree MTT",
            "et_reduction_pct": 47.4,
            "bd_rate_pct": 0.79,
        },
        {
            "authors": "Feng et al.",
            "vtm_baseline": "VTM-10.0",
            "method": "Partition map CNN",
            "et_reduction_pct": 25.70,
            "bd_rate_pct": 2.77,
        },
        {
            "authors": "Tech et al.",
            "vtm_baseline": "VTM-10.0",
            "method": "R-D-time aware CNN",
            "et_reduction_pct": 50.0,
            "bd_rate_pct": 0.7,
        },
    ]


def ra_generalization_excerpt() -> dict[str, float]:
    """Fig. 4(b) — RA first intra frame points on VTM-18.0 (paper excerpt)."""
    return {
        "pixel_ratio_pct_47bd": 47.0,
        "bd_rate_pct_47bd": 0.74,
        "pixel_ratio_pct_58bd": 58.0,
        "bd_rate_pct_58bd": 1.30,
    }


def pipeline_demo(cfg: VTMPartitionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VTMPartitionConfig()
    ctx = sample_cu_context(seed=3)
    state = extract_state_vector(ctx)
    q = predict_q_values_linear(state, seed=42)
    top3 = select_splits_top_n(q, top_n=3, threshold=cfg.rl_q_threshold)
    gt_l1, gt_l2 = sample_ground_truth_costs(seed=1)
    pred_l1 = predict_q_values_linear(state, seed=0)
    pred_l2 = predict_q_values_linear(state, seed=1)
    loss = composite_loss(
        pred_l1=pred_l1,
        gt_l1=gt_l1,
        pred_l2=pred_l2,
        gt_l2=gt_l2,
        pred_parent=min(pred_l1.values()),
        syntax_cost=12.5,
        alphas=cfg.loss_alphas,
    )
    # Toy metric: VTM-23.11 vs 18.0 CU counts (Table 1 aggregate)
    ref_cu = {qp: 1000.0 for qp in cfg.quantization_parameters}
    v2311_cu = {qp: 998.7 for qp in cfg.quantization_parameters}
    cu_ratio = complexity_ratio(ref_cu, v2311_cu, qps=cfg.quantization_parameters)
    return {
        "state_dim": int(state.size),
        "selected_splits_top3": top3,
        "composite_loss": loss,
        "cu_ratio_vtm23_vs_18": cu_ratio,
        "tradeoff_n3": rl_tradeoff_point(top_n=3, max_mtt_depth=3, cfg=cfg),
    }


def evaluation_demo(cfg: VTMPartitionConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["table_i_rows"] = len(table_i_vtm_evolution())
    demo["sota_methods"] = len(sota_partition_methods())
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_vtm_evolution": table_i_vtm_evolution(),
        "table_ii_rl_features": table_ii_rl_features(),
        "table_iii_rl_tradeoffs": table_iii_rl_tradeoffs(),
        "sota_partition_methods": sota_partition_methods(),
        "ra_generalization_excerpt": ra_generalization_excerpt(),
        "hevc_vvc_scale": hevc_vs_vvc_scale_facts(),
    }
