"""PhyWorld framework card, tables, and smoke demos (arXiv:2605.19242)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.phyworld.config import PhyWorldConfig
from ltx_trainer.phyworld.dpo import diffusion_dpo_loss, preference_margin_ok, sample_timestep_high_noise
from ltx_trainer.phyworld.flow_matching import flow_matching_loss, interpolate_latent
from ltx_trainer.phyworld.judge import overall_physics_score, round4_trainset_quotas
from ltx_trainer.phyworld.layout import LIMITATIONS
from ltx_trainer.phyworld.v2v import build_frame_mask, concat_conditioning_channels


def framework_card(cfg: PhyWorldConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PhyWorldConfig()
    return {
        "name": "PhyWorld",
        "paper": cfg.paper_arxiv,
        "hub": cfg.hub_model,
        "idea": (
            "Two-stage post-training on Wan2.2: (1) flow-matching V2V continuation for temporal "
            "consistency; (2) diffusion DPO on human physics preference pairs for law alignment."
        ),
        "stages": [
            {"stage": 1, "method": "flow_matching", "goal": "physical consistency / V2V"},
            {"stage": 2, "method": "diffusion_dpo", "goal": "physics law enforcement"},
        ],
        "base_model": cfg.base_model,
        "v2v": {"cond_frames": cfg.v2v_cond_frames, "gt_frames": cfg.v2v_gt_frames},
        "dpo": {
            "beta": cfg.dpo_beta,
            "timesteps": f"[{cfg.dpo_timestep_min}, {cfg.dpo_timestep_max}]",
            "train_pairs": cfg.dpo_train_pairs,
        },
        "evaluation": {
            "vbench_prompts": cfg.vbench_prompts,
            "physics_benchmark_prompts": cfg.benchmark_prompts,
            "judge": cfg.judge_model,
        },
        "defaults": cfg.__dict__,
    }


def table_iii_vbench() -> list[dict[str, Any]]:
    """Table 3 — VBench scores (higher is better)."""
    return [
        {
            "model": "Cosmos-2-2B",
            "subject_consistency": 0.887,
            "background_consistency": 0.905,
            "motion_smoothness": 0.964,
            "dynamic_degree": 0.543,
            "aesthetic_quality": 0.524,
            "imaging_quality": 0.608,
            "avg": 0.739,
        },
        {
            "model": "OmniWeaving",
            "subject_consistency": 0.903,
            "background_consistency": 0.907,
            "motion_smoothness": 0.972,
            "dynamic_degree": 0.556,
            "aesthetic_quality": 0.541,
            "imaging_quality": 0.621,
            "avg": 0.75,
        },
        {
            "model": "LTX-2.3-22B",
            "subject_consistency": 0.894,
            "background_consistency": 0.918,
            "motion_smoothness": 0.982,
            "dynamic_degree": 0.549,
            "aesthetic_quality": 0.532,
            "imaging_quality": 0.626,
            "avg": 0.751,
        },
        {
            "model": "Cosmos-14B",
            "subject_consistency": 0.899,
            "background_consistency": 0.923,
            "motion_smoothness": 0.973,
            "dynamic_degree": 0.559,
            "aesthetic_quality": 0.536,
            "imaging_quality": 0.629,
            "avg": 0.753,
        },
        {
            "model": "Wan2.2-I2V-A14B",
            "subject_consistency": 0.912,
            "background_consistency": 0.928,
            "motion_smoothness": 0.977,
            "dynamic_degree": 0.554,
            "aesthetic_quality": 0.543,
            "imaging_quality": 0.622,
            "avg": 0.756,
        },
        {
            "model": "PhyWorld (ours)",
            "subject_consistency": 0.932,
            "background_consistency": 0.944,
            "motion_smoothness": 0.986,
            "dynamic_degree": 0.564,
            "aesthetic_quality": 0.555,
            "imaging_quality": 0.632,
            "avg": 0.769,
        },
    ]


def table_iv_physics_faithfulness() -> list[dict[str, Any]]:
    """Table 4 — TI2V physics judge scores (1–5, higher better)."""
    return [
        {
            "model": "PhyWorld (ours)",
            "SA": 2.78,
            "PTV": 3.07,
            "Persist.": 3.23,
            "Solid-Body": 2.84,
            "Fluid": 3.04,
            "Optical": 3.57,
            "Overall": 3.09,
        },
        {
            "model": "Wan2.2-I2V-A14B",
            "SA": 2.72,
            "PTV": 2.97,
            "Persist.": 3.08,
            "Solid-Body": 2.79,
            "Fluid": 3.03,
            "Optical": 3.36,
            "Overall": 2.99,
        },
        {
            "model": "Cosmos-14B",
            "SA": 2.60,
            "PTV": 2.73,
            "Persist.": 3.07,
            "Solid-Body": 2.72,
            "Fluid": 2.92,
            "Optical": 3.53,
            "Overall": 2.80,
        },
        {
            "model": "OmniWeaving",
            "SA": 2.68,
            "PTV": 2.73,
            "Persist.": 2.92,
            "Solid-Body": 2.71,
            "Fluid": 2.99,
            "Optical": 3.13,
            "Overall": 2.78,
        },
    ]


def preference_pipeline_summary() -> dict[str, Any]:
    """Table 1 funnel excerpt."""
    return {
        "T1_retained_pairs": 3324,
        "heldout_pairs": 579,
        "T3_rl_pool_pairs": 2202,
        "trainset_pairs": 1000,
        "class_quotas": round4_trainset_quotas(),
    }


def pipeline_demo(cfg: PhyWorldConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or PhyWorldConfig()
    rng = np.random.default_rng(seed)
    dim = 8
    x0 = rng.standard_normal((dim, dim))
    x1 = rng.standard_normal((dim, dim))
    t = 0.5
    xt = interpolate_latent(x0, x1, t)
    pred_v = x1 - x0 + 0.05 * rng.standard_normal(x0.shape)
    fm_loss = flow_matching_loss(pred_v, x0, x1)

    dpo_loss = diffusion_dpo_loss(
        mse_policy_winner=0.2,
        mse_policy_loser=0.8,
        mse_ref_winner=0.3,
        mse_ref_loser=0.7,
        beta=cfg.dpo_beta,
    )

    mask = build_frame_mask(cfg.v2v_cond_frames, cfg.v2v_gt_frames, latent_h=4, latent_w=4)
    channels = concat_conditioning_channels(
        noise_latent=np.zeros((16, 4, 4)),
        condition_latent=np.ones((16, 4, 4)),
        mask=mask[: cfg.v2v_gt_frames],
    )

    return {
        "flow_matching_loss": fm_loss,
        "dpo_loss": dpo_loss,
        "preference_margin_ok": preference_margin_ok(12.0, 10.5),
        "dpo_timestep": sample_timestep_high_noise(rng, t_min=cfg.dpo_timestep_min, t_max=cfg.dpo_timestep_max),
        "v2v_channel_count": int(channels.shape[0]),
        "overall_score_toy": overall_physics_score(2.78, 3.07, 3.23, 2.84, 3.04, 3.57),
    }


def evaluation_demo(cfg: PhyWorldConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["vbench_avg"] = 0.769
    demo["physics_overall"] = 3.09
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    vbench = table_iii_vbench()
    phy = table_iv_physics_faithfulness()
    ours_v = next(r for r in vbench if "PhyWorld" in r["model"])
    base_v = next(r for r in vbench if r["model"] == "Wan2.2-I2V-A14B")
    ours_p = next(r for r in phy if "PhyWorld" in r["model"])
    base_p = next(r for r in phy if r["model"] == "Wan2.2-I2V-A14B")
    return {
        "table_iii_vbench": vbench,
        "table_iv_physics": phy,
        "preference_pipeline": preference_pipeline_summary(),
        "headline": {
            "vbench_avg_gain": ours_v["avg"] - base_v["avg"],
            "physics_overall_gain": ours_p["Overall"] - base_p["Overall"],
        },
    }
