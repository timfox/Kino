"""Framework card, VGGSound tables, evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.audioim.config import AudioImConfig
from ltx_trainer.audioim.masking import masking_demo
from ltx_trainer.audioim.style import style_conditioning_demo


def framework_card(cfg: AudioImConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioImConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "framework": cfg.framework,
        "backbone": cfg.backbone,
        "dataset": cfg.dataset,
        "demo_url": cfg.demo_url,
        "components": [
            "masked_latent_prompt_conditioning",
            "dual_timbre_tempo_encoders",
            "global_and_frame_level_adaln",
        ],
        "training": {
            "mask_ratio": "3:5",
            "prompt_s": cfg.prompt_seconds,
            "target_s": cfg.target_seconds,
            "cfg_dropout": cfg.cfg_dropout,
        },
        "encoders": {
            "timbre": cfg.timbre_encoder,
            "tempo": cfg.tempo_encoder,
            "tempo_codebooks": cfg.tempo_codebooks,
        },
        "headline": {
            "kl_panns": cfg.kl_panns,
            "ib_score": cfg.ib_score,
            "ss_mos": cfg.ss_mos,
        },
    }


def table1_v2a_performance() -> list[dict[str, Any]]:
    """Table 1 — VGGSound V2A quality."""
    return [
        {
            "method": "MMAudio (Vanilla)",
            "kl_panns": 1.72,
            "kl_passt": 1.50,
            "ib_score": 31.75,
            "desync": 0.53,
        },
        {
            "method": "MMAudio w/ Prompt Masking",
            "kl_panns": 1.71,
            "kl_passt": 1.54,
            "ib_score": 30.08,
            "desync": 0.50,
        },
        {
            "method": "AudioIM w/o Style Enc",
            "kl_panns": 1.70,
            "kl_passt": 1.49,
            "ib_score": 31.21,
            "desync": 0.51,
        },
        {
            "method": "AudioIM (ours)",
            "kl_panns": 1.65,
            "kl_passt": 1.44,
            "ib_score": 31.98,
            "desync": 0.49,
        },
    ]


def table2_style_similarity() -> list[dict[str, Any]]:
    """Table 2 — reference style similarity."""
    return [
        {
            "method": "MMAudio (Vanilla)",
            "kl_panns": 1.95,
            "kl_passt": 1.72,
            "ss_mos": 3.22,
            "ss_mos_ci": 0.24,
        },
        {
            "method": "MMAudio w/ Prompt Masking",
            "kl_panns": 1.89,
            "kl_passt": 1.71,
            "ss_mos": 3.59,
            "ss_mos_ci": 0.25,
        },
        {
            "method": "AudioIM w/o Style Enc",
            "kl_panns": 1.90,
            "kl_passt": 1.68,
            "ss_mos": 3.63,
            "ss_mos_ci": 0.28,
        },
        {
            "method": "AudioIM (ours)",
            "kl_panns": 1.85,
            "kl_passt": 1.63,
            "ss_mos": 4.06,
            "ss_mos_ci": 0.21,
        },
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_v2a_performance": table1_v2a_performance(),
        "table2_style_similarity": table2_style_similarity(),
    }


def headline_results(cfg: AudioImConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioImConfig()
    ours_t1 = next(r for r in table1_v2a_performance() if "ours" in r["method"])
    ours_t2 = next(r for r in table2_style_similarity() if "ours" in r["method"])
    vanilla_t2 = next(r for r in table2_style_similarity() if "Vanilla" in r["method"])
    return {
        "best_kl_panns": ours_t1["kl_panns"],
        "best_ib_score": ours_t1["ib_score"],
        "best_ss_mos": ours_t2["ss_mos"],
        "ss_mos_gain_vs_vanilla": round(ours_t2["ss_mos"] - vanilla_t2["ss_mos"], 2),
    }


def evaluation_demo(*, seed: int = 0, cfg: AudioImConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioImConfig()
    return {
        "framework": framework_card(cfg),
        "masking": masking_demo(seed=seed, cfg=cfg),
        "style": style_conditioning_demo(seed=seed, cfg=cfg),
        "headline": headline_results(cfg),
    }
