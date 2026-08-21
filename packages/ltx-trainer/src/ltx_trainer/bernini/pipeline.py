"""Bernini framework card, benchmarks, and smoke demos (arXiv:2605.22344)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.bernini.config import BerniniConfig
from ltx_trainer.bernini.layout import LIMITATIONS
from ltx_trainer.bernini.mock import toy_segment_ids
from ltx_trainer.bernini.objectives import total_objective
from ltx_trainer.bernini.planner import inference_mask_ratio, train_mask_ratio_beta, visible_token_fraction
from ltx_trainer.bernini.renderer import guidance_scales_t2v, guidance_scales_v2v, incremental_guidance_prediction
from ltx_trainer.bernini.sa_rope import sa_3d_rope_modulated_phase


def framework_card(cfg: BerniniConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BerniniConfig()
    return {
        "name": "Bernini",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_url,
        "idea": (
            "MLLM planner predicts target semantics in ViT embedding space (masked generative refinement); "
            "DiT renderer runs flow matching in VAE latents conditioned on semantic z, text, and source VAE "
            "features. SA-3D RoPE disambiguates multi-segment 3D positions; CoT augments the planner."
        ),
        "planner": cfg.planner_backbone,
        "renderer": cfg.renderer_backbone,
        "inference": {
            "planning_steps": cfg.planning_steps_K,
            "vit_decoder_flow_steps": cfg.vit_decoder_flow_steps,
            "dit_steps_t2v": cfg.dit_steps_t2v,
            "dit_steps_edit": cfg.dit_steps_edit,
            "flow_shift": cfg.flow_shift_dit,
        },
        "pretrain_scale": {
            "video_pairs_M": cfg.video_pair_pretrain_M,
            "image_pairs_M": cfg.image_pair_pretrain_M,
        },
    }


def bernini_bench_card() -> dict[str, Any]:
    """Sec. 6.2 — Bernini-Bench scope."""
    return {
        "cases": 300,
        "v2v_categories": 22,
        "rv2v_overlap_categories": 8,
        "metrics": ["IF", "VC", "IC", "GQ", "OS"],
        "evaluation_modes": ["MLLM_scoring", "human_SBS"],
    }


def table_bt_video_editing_leaderboard() -> list[dict[str, str | float]]:
    """Fig. 1(A) — Bradley–Terry scores (geom. mean anchor 1000)."""
    return [
        {"method": "HappyHorse-1.0", "bt_score": 1080.0, "win_pct": 61.3},
        {"method": "Bernini", "bt_score": 1044.0, "win_pct": 56.3},
        {"method": "Wan2.7", "bt_score": 1034.0, "win_pct": 54.9},
        {"method": "Grok-imagine-video", "bt_score": 964.0, "win_pct": 44.9},
        {"method": "Kling_v3_omni (std)", "bt_score": 878.0, "win_pct": 33.1},
    ]


def table_bernini_bench_mllm() -> list[dict[str, str | float]]:
    """Table 6 — Bernini-V2V / Bernini-RV2V (OS, IF, VC, GQ; RV2V adds IC)."""
    return [
        {"method": "UniVideo", "v2v_os": 2.44, "rv2v_os": 2.36},
        {"method": "VINO", "v2v_os": 2.85, "rv2v_os": 2.25},
        {"method": "Kling O3", "v2v_os": 3.05, "rv2v_os": 3.14},
        {"method": "Wan2.7", "v2v_os": 3.30, "rv2v_os": 3.58},
        {"method": "Bernini", "v2v_os": 3.49, "rv2v_os": 3.50, "v2v_vc": 3.51, "rv2v_vc": 3.51},
    ]


def table_openve_bench() -> list[dict[str, str | float]]:
    """Table 7 — OpenVE-Bench overall and Bernini."""
    return [
        {"method": "VACE-14B", "overall": 1.57},
        {"method": "OpenVE-Edit", "overall": 2.49},
        {"method": "VINO", "overall": 3.18},
        {"method": "Bernini", "overall": 4.04},
    ]


def table_editverse() -> list[dict[str, str | float]]:
    """Table 8 — editing quality / pick score."""
    return [
        {"method": "EditVerse", "editing_quality": 7.65, "pick_score": 20.07},
        {"method": "Bernini", "editing_quality": 8.02, "pick_score": 20.26},
    ]


def table_vbench_total() -> list[dict[str, str | float]]:
    """Table 11 — VBench total score."""
    return [
        {"method": "Wan2.2-A14B", "total": 84.79},
        {"method": "Bernini", "total": 84.64},
    ]


def table_opens2v_eval() -> list[dict[str, str | float]]:
    """Table 12 — OpenS2V-Eval total / FaceSim."""
    return [
        {"method": "RefAlign-14B", "total": 60.42, "facesim": 55.23},
        {"method": "Kling O3", "total": 59.19, "facesim": 57.20},
        {"method": "Bernini", "total": 62.94, "facesim": 78.20},
    ]


def table_reasoning_ablation_os() -> list[dict[str, float]]:
    """Table 10 — overall score on Bernini-V2V."""
    return [
        {"variant": "Ours (baseline)", "os": 3.12},
        {"variant": "+ PE (Qwen2.5-VL-7B)", "os": 3.20},
        {"variant": "+ Self-text", "os": 3.33},
        {"variant": "+ PE (GPT-5.4)", "os": 3.49},
        {"variant": "+ PE (GPT-5.4) + Self-visual-text", "os": 3.52},
    ]


def pipeline_demo(cfg: BerniniConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BerniniConfig()
    m0 = inference_mask_ratio(0, cfg.planning_steps_K)
    m_last = inference_mask_ratio(cfg.planning_steps_K - 1, cfg.planning_steps_K)
    phase0 = sa_3d_rope_modulated_phase(0, 0, 0, segment_index=0)
    phase1 = sa_3d_rope_modulated_phase(0, 0, 0, segment_index=1)
    seg_ids = toy_segment_ids(12, num_segments=3)
    scales = guidance_scales_v2v()
    eps_hat = incremental_guidance_prediction(
        0.1,
        0.15,
        0.2,
        0.35,
        0.5,
        omega_vid=float(scales["omega_vid"] or 0.0),
        omega_img=float(scales["omega_img"] or 0.0),
        omega_txt=float(scales["omega_txt"] or 0.0),
        omega_tgt=float(scales["omega_tgt"] or 0.0),
    )
    total = total_objective(0.5, 0.2, 0.3, lambda_text=cfg.lambda_text, lambda_visual=cfg.lambda_visual, lambda_dit=cfg.lambda_dit)
    return {
        "mask_ratio_step0": m0,
        "mask_ratio_final_step": m_last,
        "mask_monotonic_decrease": m0 > m_last,
        "train_mask_sample": train_mask_ratio_beta(alpha=5.0, beta=1.1),
        "visible_after_train_mask": visible_token_fraction(train_mask_ratio_beta()),
        "sa_rope_phase_delta_seg0_vs_seg1": abs(phase1 - phase0),
        "incremental_guidance_scalar": eps_hat,
        "total_loss_scalar": total,
        "segment_id_span": max(seg_ids) - min(seg_ids),
    }


def evaluation_demo(cfg: BerniniConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BerniniConfig()
    bt = table_bt_video_editing_leaderboard()
    bern = next(r for r in bt if r["method"] == "Bernini")
    wan = next(r for r in bt if r["method"] == "Wan2.7")
    return {
        "framework": framework_card(cfg),
        "bernini_bench": bernini_bench_card(),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "bt_delta_vs_wan2_7": round(float(bern["bt_score"]) - float(wan["bt_score"]), 1),
        "paper_tables": {
            "bt_editing_leaderboard": bt,
            "bernini_bench_mllm": table_bernini_bench_mllm(),
            "openve": table_openve_bench(),
            "editverse": table_editverse(),
            "vbench": table_vbench_total(),
            "opens2v": table_opens2v_eval(),
            "reasoning_ablation_os": table_reasoning_ablation_os(),
        },
    }
