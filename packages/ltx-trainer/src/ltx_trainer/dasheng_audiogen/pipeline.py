"""Dasheng AudioGen framework card and benchmark bundles."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dasheng_audiogen.captions import CAPTION_VIEWS, caption_smoke
from ltx_trainer.dasheng_audiogen.config import DashengAudioGenConfig
from ltx_trainer.dasheng_audiogen.eval import eval_smoke
from ltx_trainer.dasheng_audiogen.flow import flow_smoke
from ltx_trainer.dasheng_audiogen.layout import LIMITATIONS
from ltx_trainer.dasheng_audiogen.metrics import (
    mecat_category_legend,
    table_1_capability_comparison,
    table_2_standard_benchmarks,
    table_3_mecat_mixed,
    table_4_structured_ablation,
    unified_vs_acoustic_gains,
)


def framework_card(cfg: DashengAudioGenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DashengAudioGenConfig()
    return {
        "name": "Dasheng AudioGen",
        "paper": cfg.paper_arxiv,
        "demo_url": cfg.demo_url,
        "idea": (
            "Unified flow-matching DiT for coherent mixed-audio scenes from text: structured "
            "multi-view captions plus DashengTokenizer semantic-acoustic latents (1280-d @ 25 Hz)."
        ),
        "architecture": {
            "dit_layers": cfg.dit_layers,
            "dit_hidden": cfg.dit_hidden,
            "dit_params_b": cfg.dit_params_b,
            "text_encoder": cfg.text_encoder,
            "latent_dim": cfg.latent_dim,
            "latent_hz": cfg.latent_hz,
            "clip_duration_s": cfg.clip_duration_s,
            "fm_steps": cfg.fm_steps,
            "cfg_scale": cfg.cfg_scale,
        },
        "caption_views": list(CAPTION_VIEWS),
        "training": {
            "dataset": "ACAVCaps superset",
            "hours": cfg.train_hours,
            "steps": cfg.train_steps,
        },
        "unique_capability": "Only model with SFX + music + intelligible speech + audio scene (Table 1)",
        "limitations": list(LIMITATIONS),
    }


def headline_results(cfg: DashengAudioGenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DashengAudioGenConfig()
    return {
        "mixed_audio": (
            f"SMA FAD {cfg.mecat_sma_fad} vs Expert-Pipeline {cfg.expert_pipeline_sma_fad}; "
            f"WER {cfg.mecat_sma_wer_pct}% vs {cfg.expert_pipeline_sma_wer_pct}%"
        ),
        "single_type": f"MusicCaps FAD {cfg.musiccaps_fad}; LibriTTS UTMOS {cfg.librispeech_utmos} (WER {cfg.librispeech_wer_pct}%)",
        "structured_captions": f"LibriTTS WER {cfg.librispeech_wer_pct}% vs unstructured {cfg.unstructured_wer_pct}%",
        "representation": "Unified DashengTokenizer beats acoustic VAE ~20% avg on MECAT (ACAVCaps training)",
        "pafi_sma": f"PAFI {cfg.pafi_ours_sma} tied with GT {cfg.pafi_gt_sma}, above Expert {cfg.pafi_expert_sma}",
    }


def pipeline_demo(cfg: DashengAudioGenConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or DashengAudioGenConfig()
    return {
        "captions": caption_smoke(),
        "flow": flow_smoke(cfg, seed=seed),
        "eval": eval_smoke(cfg),
    }


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    return {
        "headline": headline_results(),
        "demo": pipeline_demo(seed=seed),
        "framework": framework_card(),
    }


def benchmarks_bundle(cfg: DashengAudioGenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DashengAudioGenConfig()
    return {
        "table_1_capability": table_1_capability_comparison(),
        "table_2_standard": table_2_standard_benchmarks(cfg),
        "table_3_mecat_mixed": table_3_mecat_mixed(cfg),
        "table_4_structured_ablation": table_4_structured_ablation(cfg),
        "unified_vs_acoustic_gains": unified_vs_acoustic_gains(),
        "mecat_categories": mecat_category_legend(),
        "headline": headline_results(cfg),
    }
