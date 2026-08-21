"""LMDM framework card, paper tables, and smoke demos (arXiv:2605.22717)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lmdm.arc_forcing import arc_forcing_total
from ltx_trainer.lmdm.complexity import lmdm_block_causal_ops, lmdm_enc_dec_ops, lmm_decode_ops
from ltx_trainer.lmdm.config import LMDMConfig, LMDMLatencyStats, LMDMVariant
from ltx_trainer.lmdm.layout import LIMITATIONS
from ltx_trainer.lmdm.mock import toy_discriminator_scores
from ltx_trainer.lmdm.routing import context_hidden_independent_of_noise, routing_mask


def framework_card(cfg: LMDMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LMDMConfig()
    lat = LMDMLatencyStats()
    return {
        "name": "LMDM",
        "paper": cfg.paper_arxiv,
        "demo": cfg.demo_url,
        "idea": (
            "Repurpose open-source audio diffusion (flow matching) into interactive streaming "
            "music via clean/noisy routing + attention masks for KV-caching; ARC-Forcing for "
            "RL-free rollout post-training."
        ),
        "variants": [LMDMVariant.ENCODER_DECODER.value, LMDMVariant.BLOCK_CAUSAL.value],
        "backbone": {"name": cfg.backbone, "params_M": cfg.backbone_params_M},
        "streaming": {
            "context_latent_frames": cfg.context_s,
            "block_latent_frames": cfg.block_size_o,
            "total_latent_frames": cfg.latent_frames_total,
        },
        "controls": ["global_text", "sketch_cqt_loudness", "accompaniment_stem"],
        "latency_ms": {"pre_arc": lat.pre_arc_ms, "post_arc": lat.post_arc_ms},
    }


def table_text_global() -> list[dict[str, str | float | int | bool]]:
    """Table 1 — text-conditioned LMDM vs baselines."""
    rows = [
        {"method": "Magenta RealTime", "d_nfe": 800, "ttff_s": 4.0, "fd": 72.14, "kl": 0.47, "clap": 0.35},
        {"method": "Stable Audio Open", "d_nfe": 100, "ttff_s": 10.35, "fd": 96.51, "kl": 0.55, "clap": 0.41},
        {"method": "MusicGen-Large", "d_nfe": 2400, "ttff_s": 10.81, "fd": 190.47, "kl": 0.52, "clap": 0.31},
        {"method": "LMDM (ED)", "d_nfe": 50, "ttff_s": 0.11, "fd": 61.06, "kl": 1.14, "clap": 0.20},
        {"method": "LMDM (ED)+AF", "d_nfe": 8, "ttff_s": 0.03, "fd": 35.88, "kl": 0.74, "clap": 0.29},
        {"method": "LMDM (BC)+AF", "d_nfe": 2, "ttff_s": 0.02, "fd": 47.26, "kl": 0.91, "clap": 0.23},
        {"method": "LMDM (ED)+AF primed", "d_nfe": 8, "ttff_s": 0.03, "fd": 29.00, "kl": 0.35, "clap": 0.32},
    ]
    return rows


def table_sketch_musdb() -> list[dict[str, float | str | int | bool]]:
    """Table 2 — sketch-conditioned MUSDB18 rollouts."""
    return [
        {"method": "LMDM (ED)", "d_nfe": 50, "blocks": 5, "af": False, "fd": 101.01, "mel": 0.26, "dyn": 0.46},
        {"method": "LMDM (ED)+AF", "d_nfe": 8, "blocks": 5, "af": True, "fd": 181.79, "mel": 0.27, "dyn": 0.45},
        {"method": "Bidir Flow", "d_nfe": 50, "blocks": 1, "af": False, "fd": 78.51, "mel": 0.33, "dyn": 0.57},
    ]


def design_space_axes() -> dict[str, list[str]]:
    """Sec. 5 — conditioning axes for live music agents."""
    return {
        "scope": ["global", "local"],
        "interaction": ["instrument_like", "accompaniment_like"],
        "examples": [
            "global_text_prompt_transitions",
            "sketch_cqt_loudness",
            "stem_accompaniment_jamming",
        ],
    }


def pipeline_demo(cfg: LMDMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LMDMConfig()
    mask = routing_mask(cfg.context_s, cfg.block_size_o)
    ctx_hidden = context_hidden_independent_of_noise(1.0, w_clean=0.8)
    scores = toy_discriminator_scores()
    arc_loss = arc_forcing_total(
        scores["rollout"],
        scores["real"],
        d_matched=scores["matched"],
        d_mismatched=scores["mismatched"],
    )
    return {
        "routing_mask_len": len(mask),
        "routing_target_ones": sum(mask),
        "context_hidden": ctx_hidden,
        "complexity_ops": {
            "lmm": lmm_decode_ops(cfg.context_s, cfg.block_size_o),
            "lmdm_enc_dec": lmdm_enc_dec_ops(cfg.context_s, cfg.block_size_o, cfg.diffusion_steps_K),
            "lmdm_block_causal": lmdm_block_causal_ops(cfg.context_s, cfg.block_size_o, cfg.diffusion_steps_K),
        },
        "arc_forcing_loss_scalar": arc_loss,
    }


def evaluation_demo(cfg: LMDMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LMDMConfig()
    primed = next(r for r in table_text_global() if r["method"] == "LMDM (ED)+AF primed")
    musicgen = next(r for r in table_text_global() if r["method"] == "MusicGen-Large")
    ttff_speedup = float(musicgen["ttff_s"]) / float(primed["ttff_s"])
    return {
        "framework": framework_card(cfg),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "ttff_speedup_vs_musicgen_large": round(ttff_speedup, 1),
        "paper_tables": {
            "text_global": table_text_global(),
            "sketch_musdb": table_sketch_musdb(),
            "design_space": design_space_axes(),
        },
    }
