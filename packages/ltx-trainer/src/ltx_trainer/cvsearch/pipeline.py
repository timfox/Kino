"""CVSearch framework card, paper tables, and smoke demos (ICML 2026)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cvsearch.config import CVSearchConfig, SearchMode, adaptive_tree_depth
from ltx_trainer.cvsearch.layout import LIMITATIONS
from ltx_trainer.cvsearch.mock import toy_existence_confidences, toy_tree_layer
from ltx_trainer.cvsearch.search import (
    bottom_up_search_layer,
    expert_coverage_valid,
    route_search_mode,
)
from ltx_trainer.cvsearch.sgap import clustering_cost, select_optimal_k, visual_complexity


def framework_card(cfg: CVSearchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CVSearchConfig()
    return {
        "name": "CVSearch",
        "paper": cfg.paper_arxiv,
        "venue": cfg.venue,
        "code": cfg.code_repo,
        "idea": (
            "Training-free Assess-then-Search for HR MLLMs: global sufficiency → SAM 3 expert proposals → "
            "Scene-aware Scanning (SGAP + Dynamic Bottom-Up Search) on expert failure."
        ),
        "workflow": [
            "Global cognitive assessment (cq vs τq)",
            "Visual Expert Assisted Search (SAM 3 + query parsing)",
            "Scene-aware Scanning (SGAP tree + bottom-up priority)",
        ],
        "thresholds": {
            "tau_q": cfg.tau_q,
            "tau_q_hat": cfg.tau_q_hat,
            "tau_v": cfg.tau_v,
        },
        "sgap": {"k_min": cfg.k_min, "k_max": cfg.k_max},
        "priority_weights": {"alpha": cfg.alpha_cv, "beta": cfg.beta_co, "gamma": cfg.gamma_child},
        "tree_depth": {"single_object": cfg.depth_single_object, "multi_object": cfg.depth_multi_object},
        "visual_expert": cfg.visual_expert,
        "default_backbone": cfg.default_backbone,
    }


def table_hr_benchmarks() -> dict[str, dict[str, float | str]]:
    """Table 1 excerpt — CVSearch gains on Qwen2.5-VL-7B and LLaVA-OV-7B."""
    return {
        "LLaVA-OV-7B": {"V*_Overall": 75.4, "HR4K_Overall": 63.0, "HR8K_Overall": 59.8},
        "LLaVA-OV-7B + CVSearch": {"V*_Overall": 91.6, "HR4K_Overall": 75.6, "HR8K_Overall": 74.8},
        "Qwen2.5-VL-7B": {"V*_Overall": 71.2, "HR4K_Overall": 68.8, "HR8K_Overall": 65.3},
        "Qwen2.5-VL-7B + CVSearch": {"V*_Overall": 90.1, "HR4K_Overall": 76.6, "HR8K_Overall": 75.6},
        "InternVL2.5-8B": {"V*_Overall": 69.1, "HR4K_Overall": 66.0, "HR8K_Overall": 57.4},
        "InternVL2.5-8B + CVSearch": {"V*_Overall": 89.0, "HR4K_Overall": 77.0, "HR8K_Overall": 77.6},
    }


def table_visual_search_comparison() -> dict[str, dict[str, float]]:
    """Table 2 — vs SEAL, DyFo, ZoomEye, RAP."""
    return {
        "SEAL": {"V*": 75.4, "HR4K": 0.0, "HR8K": 0.0},
        "DyFo": {"V*": 81.2, "HR4K": 0.0, "HR8K": 0.0},
        "LLaVA-OV-7B + ZoomEye": {"V*": 90.6, "HR4K": 69.6, "HR8K": 69.3},
        "LLaVA-OV-7B + RAP": {"V*": 79.6, "HR4K": 71.0, "HR8K": 67.6},
        "LLaVA-OV-7B + CVSearch": {"V*": 91.6, "HR4K": 75.6, "HR8K": 74.8},
        "Qwen2.5-VL-7B + CVSearch": {"V*": 90.1, "HR4K": 76.6, "HR8K": 75.6},
        "InternVL2.5-8B + CVSearch": {"V*": 89.0, "HR4K": 77.0, "HR8K": 77.6},
    }


def table_general_benchmarks() -> dict[str, dict[str, float | str]]:
    """Table 3 excerpt."""
    return {
        "LLaVA-OV-7B": {"MME_RW_Lite": 43.7, "TreeBench": 37.3, "FineRS4K_MVQA": 72.0, "FineRS4K_OVQA": 49.7},
        "LLaVA-OV-7B + CVSearch": {"MME_RW_Lite": 48.8, "TreeBench": 38.8, "FineRS4K_MVQA": 77.4, "FineRS4K_OVQA": 61.2},
        "Qwen2.5-VL-7B + CVSearch": {"MME_RW_Lite": 46.7, "TreeBench": 40.7, "FineRS4K_MVQA": 82.5, "FineRS4K_OVQA": 58.3},
    }


def table_throughput() -> dict[str, dict[str, float]]:
    """Table 5 — accuracy and throughput (samples/min, Qwen2.5-VL-7B)."""
    return {
        "Qwen2.5-VL-7B": {"V*_Acc": 71.2, "V*_Thr": 8.30, "HR4K_Acc": 68.8, "HR4K_Thr": 7.62, "HR8K_Acc": 65.3, "HR8K_Thr": 7.62},
        "SAM 3 expert only": {"V*_Acc": 84.3, "V*_Thr": 3.60, "HR4K_Acc": 71.9, "HR4K_Thr": 5.59, "HR8K_Acc": 68.1, "HR8K_Thr": 5.30},
        "ZoomEye": {"V*_Acc": 85.3, "V*_Thr": 0.68, "HR4K_Acc": 72.5, "HR4K_Thr": 1.29, "HR8K_Acc": 69.8, "HR8K_Thr": 0.68},
        "RAP": {"V*_Acc": 84.8, "V*_Thr": 0.66, "HR4K_Acc": 74.8, "HR4K_Thr": 1.22, "HR8K_Acc": 76.0, "HR8K_Thr": 0.58},
        "CVSearch": {"V*_Acc": 90.1, "V*_Thr": 1.02, "HR4K_Acc": 76.6, "HR4K_Thr": 3.77, "HR8K_Acc": 75.6, "HR8K_Thr": 1.92},
    }


def table_ablation() -> dict[str, dict[str, float]]:
    """Table 6 — progressive ablation on Qwen2.5-VL-7B."""
    return {
        "Baseline": {"V*": 71.2, "HR4K": 68.8, "HR8K": 65.3, "HR4K_Thr": 7.62},
        "+ SAM 3": {"V*": 84.3, "HR4K": 71.9, "HR8K": 68.1, "HR4K_Thr": 5.59},
        "+ Rigid Grid + Top-down": {"V*": 84.8, "HR4K": 73.5, "HR8K": 70.3, "HR4K_Thr": 2.02},
        "+ SGAP + Top-down": {"V*": 84.8, "HR4K": 72.3, "HR8K": 70.8, "HR4K_Thr": 2.14},
        "+ SGAP + Bottom-up": {"V*": 86.4, "HR4K": 76.8, "HR8K": 74.9, "HR4K_Thr": 3.81},
        "CVSearch (full)": {"V*": 90.1, "HR4K": 76.6, "HR8K": 75.6, "HR4K_Thr": 3.77},
    }


def figure1_highlights() -> dict[str, float]:
    """Fig. 1(c) headline deltas on Qwen2.5-VL-7B backbone."""
    return {"accuracy_gain_vstar_pp": 4.8, "accuracy_gain_hr4k_pp": 4.1, "throughput_vs_zoom_eye_x": 2.9, "throughput_vs_rap_x": 1.5}


def sgap_demo(cfg: CVSearchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CVSearchConfig()
    feats = [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]]
    labels = [0, 0, 1, 1]
    boxes = [(0, 0, 1, 1), (1, 0, 1, 1)]
    k_star = select_optimal_k(k_min=cfg.k_min, k_max=cfg.k_max, candidate_costs={4: 0.2, 5: 0.15, 6: 0.18})
    return {
        "k_star": k_star,
        "clustering_cost_k4": clustering_cost(4, boxes, feats, labels),
        "visual_complexity_foreground": visual_complexity(feats[:2]),
        "visual_complexity_background": visual_complexity([[0.01, 0.01], [0.02, 0.01]]),
    }


def routing_demo(cfg: CVSearchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CVSearchConfig()
    return {
        "direct": route_search_mode(0.95, True, True, tau_q=cfg.tau_q).value,
        "expert": route_search_mode(0.5, True, expert_coverage_valid(2, 2), tau_q=cfg.tau_q).value,
        "scan": route_search_mode(0.5, False, False, tau_q=cfg.tau_q).value,
        "depth_m1": adaptive_tree_depth(1, cfg),
        "depth_m2": adaptive_tree_depth(2, cfg),
    }


def bottom_up_demo(cfg: CVSearchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CVSearchConfig()
    nodes = [n for n in toy_tree_layer() if not n.pruned]
    conf = toy_existence_confidences()
    result, steps = bottom_up_search_layer(nodes, conf, cfg=cfg)
    return {
        "found": result.found,
        "candidate": result.candidate_node_id,
        "sufficiency": result.sufficiency,
        "search_steps": steps,
    }


def evaluation_demo(cfg: CVSearchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CVSearchConfig()
    return {
        "framework": framework_card(cfg),
        "sgap": sgap_demo(cfg),
        "routing": routing_demo(cfg),
        "bottom_up": bottom_up_demo(cfg),
        "limitations": list(LIMITATIONS),
        "paper_tables": {
            "hr_benchmarks": table_hr_benchmarks(),
            "visual_search": table_visual_search_comparison(),
            "general": table_general_benchmarks(),
            "throughput": table_throughput(),
            "ablation": table_ablation(),
            "figure1": figure1_highlights(),
        },
    }
