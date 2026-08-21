"""CounterFlow framework card, tables, and smoke demos (arXiv:2605.18916)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.counterflow.config import CounterFlowConfig
from ltx_trainer.counterflow.guidance import euler_sample_counterflow, vanilla_cfg_velocity
from ltx_trainer.counterflow.layout import LIMITATIONS
from ltx_trainer.counterflow.metrics import delta_flam, p_flam_max_frame, positive_delta_flam_ratio
from ltx_trainer.counterflow.mock import LinearVelocityProvider


def framework_card(cfg: CounterFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CounterFlowConfig()
    return {
        "name": "CounterFlow",
        "paper": cfg.paper_arxiv,
        "demo": cfg.demo_url,
        "task": "Counterfactual Video Foley Generation (VT2A)",
        "idea": (
            "Two-phase inference on pretrained flow-matching VT2A: Phase 1 uses decomposed "
            "video+text guidance to build temporal structure while suppressing visually implied source; "
            "Phase 2 drops video and applies target-vs-source text CFG for counterfactual timbre."
        ),
        "phase1": "Eq. (1) — w_vid video term + w_txt (target − source) text contrast",
        "phase2": "Eq. (2) — w_cfg text-only (target − source), video null",
        "backbone": cfg.backbone,
        "hyperparameters": {
            "N": cfg.num_steps,
            "N_trans": cfg.transition_step,
            "w_vid": cfg.guidance_w_vid,
            "w_txt": cfg.guidance_w_txt,
            "w_cfg": cfg.guidance_w_cfg,
        },
        "evaluation": {
            "dataset": cfg.dataset,
            "triplets": cfg.dataset_triplets,
            "metrics": ["FAD", "IS", "CLAP", "DeSync", "ΔFLAM", "positive-ΔFLAM ratio"],
        },
        "defaults": cfg.__dict__,
    }


def table_i_main_comparison() -> list[dict[str, float | str]]:
    """Table 1 — main comparison on VGGSound-Sparse Clean."""
    return [
        {
            "method": "CAFA",
            "fad": 24.81,
            "is": 5.931,
            "delta_flam": 0.1289,
            "positive_ratio": 0.8258,
            "clap": 0.2371,
            "desync": 0.5888,
        },
        {
            "method": "CAFA + neg.",
            "fad": 31.46,
            "is": 7.606,
            "delta_flam": 0.2573,
            "positive_ratio": 0.8835,
            "clap": 0.1801,
            "desync": 0.6431,
        },
        {
            "method": "ReWaS",
            "fad": 75.18,
            "is": 4.223,
            "delta_flam": 0.0560,
            "positive_ratio": 0.6184,
            "clap": 0.1084,
            "desync": 1.078,
        },
        {
            "method": "ReWaS + neg.",
            "fad": 79.52,
            "is": 4.703,
            "delta_flam": 0.1905,
            "positive_ratio": 0.7130,
            "clap": 0.0947,
            "desync": 1.103,
        },
        {
            "method": "CounterFlow",
            "fad": 23.55,
            "is": 7.915,
            "delta_flam": 0.2641,
            "positive_ratio": 0.9200,
            "clap": 0.2840,
            "desync": 0.6695,
        },
        {
            "method": "CounterFlow w/o P2 neg.",
            "fad": 23.29,
            "is": 7.790,
            "delta_flam": 0.2373,
            "positive_ratio": 0.9170,
            "clap": 0.2849,
            "desync": 0.6261,
        },
    ]


def table_ii_ablations() -> list[dict[str, float | str]]:
    """Table 2 — ablation on Phase 1 decomposition and phase swap."""
    return [
        {
            "method": "CounterFlow",
            "fad": 23.55,
            "delta_flam": 0.2641,
            "desync": 0.6695,
            "clap": 0.2840,
        },
        {
            "method": "w/o P1 decomp. CFG",
            "fad": 24.36,
            "delta_flam": 0.0278,
            "desync": 0.2390,
            "clap": 0.0894,
        },
        {
            "method": "w/o P1 neg.",
            "fad": 21.00,
            "delta_flam": 0.0534,
            "desync": 0.4362,
            "clap": 0.2608,
        },
        {
            "method": "Phase swap (P1 ↔ P2)",
            "fad": 52.33,
            "delta_flam": 0.2367,
            "desync": 0.9989,
            "clap": 0.2817,
        },
    ]


def figure3_ntrans_sweep() -> list[dict[str, float | int]]:
    """Fig. 3 — transition-step trade-off (representative points from paper)."""
    return [
        {"n_trans": 1, "delta_flam": 0.35, "desync": 1.15},
        {"n_trans": 5, "delta_flam": 0.32, "desync": 1.05},
        {"n_trans": 9, "delta_flam": 0.30, "desync": 0.95},
        {"n_trans": 13, "delta_flam": 0.28, "desync": 0.85},
        {"n_trans": 17, "delta_flam": 0.2641, "desync": 0.6695},
        {"n_trans": 21, "delta_flam": 0.22, "desync": 0.55},
        {"n_trans": 25, "delta_flam": 0.18, "desync": 0.45},
    ]


def pipeline_demo(cfg: CounterFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CounterFlowConfig()
    provider = LinearVelocityProvider(dim=16, seed=42)
    z0 = np.random.default_rng(0).standard_normal(16)
    traj = euler_sample_counterflow(
        provider,
        z0,
        num_steps=cfg.num_steps,
        transition_step=cfg.transition_step,
        w_vid=cfg.guidance_w_vid,
        w_txt=cfg.guidance_w_txt,
        w_cfg=cfg.guidance_w_cfg,
    )
    z_final = traj[-1]
    audio_stub = z_final  # latent stand-in for decoded waveform features
    p_tar = p_flam_max_frame("lion roaring", audio_stub, seed=1)
    p_src = p_flam_max_frame("dog barking", audio_stub, seed=2)
    d_flam = delta_flam(p_tar, p_src)
    v_vanilla = vanilla_cfg_velocity(provider, z0, w=cfg.guidance_w_cfg)
    return {
        "trajectory_length": len(traj),
        "phase1_steps": cfg.transition_step,
        "phase2_steps": cfg.num_steps - cfg.transition_step,
        "final_latent_norm": float(np.linalg.norm(z_final)),
        "vanilla_cfg_norm": float(np.linalg.norm(v_vanilla)),
        "delta_flam_toy": d_flam,
        "positive_ratio_toy": positive_delta_flam_ratio([d_flam, 0.1, -0.05, 0.2]),
    }


def evaluation_demo(cfg: CounterFlowConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["counterflow_table1_delta_flam"] = next(
        r["delta_flam"] for r in table_i_main_comparison() if r["method"] == "CounterFlow"
    )
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_main": table_i_main_comparison(),
        "table_ii_ablations": table_ii_ablations(),
        "figure3_ntrans_sweep": figure3_ntrans_sweep(),
    }
