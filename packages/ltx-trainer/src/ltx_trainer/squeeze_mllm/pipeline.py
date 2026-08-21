"""Squeeze-MLLM framework card, paper tables, and smoke demos (arXiv:2605.26111)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.squeeze_mllm.config import SqueezeMLLMConfig
from ltx_trainer.squeeze_mllm.denoising import mask_schedule, stage_masks
from ltx_trainer.squeeze_mllm.dla import DualLayerAggregator, synthetic_mllm_layer_stack
from ltx_trainer.squeeze_mllm.lap import LayerwiseAttentionPooling
from ltx_trainer.squeeze_mllm.layout import architecture_layout, paper_limitations


def framework_card(cfg: SqueezeMLLMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SqueezeMLLMConfig()
    return {
        "name": "Squeeze-MLLM (subject-driven generation)",
        "paper": cfg.paper_arxiv,
        "project_page": cfg.project_page,
        "idea": "Joint MLLM text+image conditioning + VAE identity; DLA aggregates all MLLM layers per modality.",
        "components": [
            "Layerwise Attention Pooling (LAP)",
            "Dual Layer Aggregator (DLA)",
            "Multi-stage timestep-aware denoising",
            "Two-stage training (MLLM-only then MLLM+VAE)",
        ],
        "defaults": {
            "tau1": cfg.tau1,
            "tau2": cfg.tau2,
            "cfg_scale": cfg.cfg_scale,
            "mllm": cfg.mllm_name,
            "dit": cfg.dit_name,
        },
    }


def table_dreambench() -> dict[str, dict[str, float]]:
    """Table 1 — DreamBench metrics (paper excerpt)."""
    return {
        "OminiControl": {"dino_i": 0.5987, "clip_i": 0.7840, "clip_t": 0.3186},
        "OmniGen2": {"dino_i": 0.7323, "clip_i": 0.8268, "clip_t": 0.3185},
        "UNO": {"dino_i": 0.7484, "clip_i": 0.8354, "clip_t": 0.3040},
        "DreamO": {"dino_i": 0.7537, "clip_i": 0.8356, "clip_t": 0.3086},
        "USO": {"dino_i": 0.7478, "clip_i": 0.8263, "clip_t": 0.3213},
        "UMO": {"dino_i": 0.7481, "clip_i": 0.8339, "clip_t": 0.3022},
        "Qwen-Image": {"dino_i": 0.7317, "clip_i": 0.8261, "clip_t": 0.3158},
        "EasyRef": {"dino_i": 0.6961, "clip_i": 0.8153, "clip_t": 0.3031},
        "Ours (MLLM only)": {"dino_i": 0.6788, "clip_i": 0.8228, "clip_t": 0.2988},
        "Ours (MLLM + VAE)": {"dino_i": 0.7482, "clip_i": 0.8443, "clip_t": 0.3010},
    }


def table_copy_paste() -> dict[str, dict[str, float]]:
    """Table 2 — pose diversity / copy-paste proxy (higher azimuth/polar = more variation)."""
    return {
        "OmniGen2": {"azimuth": 22.6, "polar": 7.0, "avg_recall_rate": 0.486},
        "DreamO": {"azimuth": 22.1, "polar": 9.6, "avg_recall_rate": 0.372},
        "USO": {"azimuth": 20.8, "polar": 9.6, "avg_recall_rate": 0.401},
        "Qwen-Image": {"azimuth": 17.6, "polar": 7.8, "avg_recall_rate": 0.460},
        "Ours": {"azimuth": 25.7, "polar": 10.4, "avg_recall_rate": 0.349},
    }


def table_reasoning_clip_t() -> dict[str, float]:
    """Table 3 — multimodal reasoning benchmark (350 samples)."""
    return {
        "UNO": 0.2851,
        "DreamO": 0.2888,
        "Qwen-Image": 0.3099,
        "Ours": 0.3208,
    }


def table_mllm_connection_ablation() -> dict[str, dict[str, float]]:
    """Table 6 — strategies for connecting MLLM features to DiT."""
    return {
        "Last layer": {"dino_i": 0.6566, "clip_i": 0.8128, "clip_t": 0.2893},
        "Last layer (blend ViT)": {"dino_i": 0.7118, "clip_i": 0.8286, "clip_t": 0.2850},
        "Single LAP 0-28": {"dino_i": 0.7524, "clip_i": 0.8502, "clip_t": 0.2878},
        "DLA (Dual LAP) 0-28": {"dino_i": 0.7482, "clip_i": 0.8443, "clip_t": 0.3010},
    }


def table_two_stage_training() -> dict[str, dict[str, float]]:
    """Table 7 — single-stage vs two-stage training."""
    return {
        "Single-stage w/o TAD": {"dino_i": 0.7184, "clip_i": 0.8245, "clip_t": 0.2971},
        "Single-stage with TAD": {"dino_i": 0.5763, "clip_i": 0.7686, "clip_t": 0.2995},
        "Two-stage": {"dino_i": 0.7482, "clip_i": 0.8443, "clip_t": 0.3010},
    }


def table_denoising_sensitivity() -> list[dict[str, float]]:
    """Table C — τ1, τ2, CFG sensitivity (gray row = paper default)."""
    return [
        {"tau1": 0.00, "tau2": 0.00, "cfg": 2.5, "dino_i": 0.6905, "clip_i": 0.8225, "clip_t": 0.3044},
        {"tau1": 0.97, "tau2": 0.90, "cfg": 2.5, "dino_i": 0.7490, "clip_i": 0.8466, "clip_t": 0.2963},
        {"tau1": 0.95, "tau2": 0.85, "cfg": 2.5, "dino_i": 0.7482, "clip_i": 0.8443, "clip_t": 0.3010},
        {"tau1": 0.85, "tau2": 0.70, "cfg": 2.5, "dino_i": 0.7282, "clip_i": 0.8376, "clip_t": 0.3034},
        {"tau1": 0.95, "tau2": 0.85, "cfg": 4.0, "dino_i": 0.7431, "clip_i": 0.8381, "clip_t": 0.3012},
    ]


def dla_demo(cfg: SqueezeMLLMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SqueezeMLLMConfig()
    b, m, c = 2, cfg.num_mllm_layers, cfg.embed_dim
    text_layers = synthetic_mllm_layer_stack(
        batch=b, num_layers=m, seq_len=8, channels=c, seed=11, modality_bias=0.3
    )
    image_layers = synthetic_mllm_layer_stack(
        batch=b, num_layers=m, seq_len=16, channels=c, seed=22, modality_bias=-0.3
    )
    dla = DualLayerAggregator(cfg)
    out = dla(text_layers, image_layers)
    return {
        "text_agg_shape": list(out.text_agg.shape),
        "image_agg_shape": list(out.image_agg.shape),
        "text_mean": float(out.text_agg.mean().item()),
        "image_mean": float(out.image_agg.mean().item()),
    }


def single_lap_vs_dla_tradeoff(cfg: SqueezeMLLMConfig | None = None) -> dict[str, Any]:
    """Illustrates Fig. 3 tradeoff: one LAP on concatenated modalities vs DLA."""
    cfg = cfg or SqueezeMLLMConfig()
    b, m, c = 1, cfg.num_mllm_layers, cfg.embed_dim
    text = synthetic_mllm_layer_stack(batch=b, num_layers=m, seq_len=6, channels=c, seed=1, modality_bias=0.5)
    image = synthetic_mllm_layer_stack(batch=b, num_layers=m, seq_len=6, channels=c, seed=2, modality_bias=-0.5)
    joint = torch.cat([text, image], dim=2)  # concat seq: single LAP path
    single_lap = LayerwiseAttentionPooling(c, cfg.num_attn_heads)
    dla = DualLayerAggregator(cfg)
    joint_out = single_lap(joint)
    dla_out = dla(text, image)
    # Proxy: text alignment vs identity — variance along sequence as toy signal
    return {
        "single_lap_seq_var": float(joint_out.var().item()),
        "dla_text_var": float(dla_out.text_agg.var().item()),
        "dla_image_var": float(dla_out.image_agg.var().item()),
        "note": "Paper: single LAP trades CLIP-T for DINO-I; DLA balances both.",
    }


def evaluation_demo(cfg: SqueezeMLLMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SqueezeMLLMConfig()
    schedule = mask_schedule(num_steps=10, cfg=cfg)
    early = stage_masks(0.99, cfg)
    mid = stage_masks(0.90, cfg)
    late = stage_masks(0.50, cfg)
    return {
        "dla": dla_demo(cfg),
        "lap_tradeoff": single_lap_vs_dla_tradeoff(cfg),
        "stage_masks": {
            "early_t0.99": {"mllm": early.use_mllm, "vae": early.use_vae},
            "middle_t0.90": {"mllm": mid.use_mllm, "vae": mid.use_vae},
            "late_t0.50": {"mllm": late.use_mllm, "vae": late.use_vae},
        },
        "mask_schedule_sample": schedule[:3] + schedule[-2:],
        "training_stages": {
            "stage1_steps": cfg.stage1_steps,
            "stage2_steps": cfg.stage2_steps,
            "stage1_conditioning": "MLLM only",
            "stage2_conditioning": "MLLM + VAE",
        },
        "limitations": paper_limitations(),
        "layout": architecture_layout(),
    }
