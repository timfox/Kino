"""PlanAudio framework card, knowledge, and evaluation demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.planaudio.benchmarks import benchmarks_bundle
from ltx_trainer.planaudio.config import PlanAudioConfig
from ltx_trainer.planaudio.cot import (
    downsample_af3_embeddings,
    format_sequence,
    latent_supervision_loss,
    total_loss,
)
from ltx_trainer.planaudio.hooks import score_prompt_composition
from ltx_trainer.planaudio.scenarios import SCENARIO_COMPOSITE, scenario_card
from ltx_trainer.planaudio.scoring import normalized_scenario_score


def framework_card(cfg: PlanAudioConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PlanAudioConfig()
    return {
        "name": "PlanAudio",
        "paper": cfg.paper_arxiv,
        "title": cfg.paper_title,
        "task": "Free-Form-Text-Prompt-to-Unified-Audio Generation",
        "architecture": {
            "backbone": cfg.backbone_llm,
            "semantic_encoder": cfg.semantic_encoder,
            "acoustic_tokenizer": cfg.acoustic_tokenizer,
            "no_external_text_encoder": True,
            "latent_cot_steps": cfg.latent_cot_steps,
        },
        "training": {
            "pool_size": cfg.training_pool_size,
            "composite": cfg.composite_train,
            "sound": cfg.sound_train,
            "speech": cfg.speech_train,
            "bench_holdout": cfg.planaudio_bench_size,
        },
        "benchmark": "PlanAudio-Bench (4.5k composite clips)",
        "vs_pipeline": "No TTS+T2S merge; end-to-end from free-form text",
        "vs_voiceldm": "No Gemini text-rewrite module required",
    }


def knowledge_card(cfg: PlanAudioConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PlanAudioConfig()
    bench = benchmarks_bundle()
    return {
        "framework": framework_card(cfg),
        "scenarios": scenario_card(),
        "key_results": {
            "composite_fad_panns_planaudio": bench["table2_composite"]["PlanAudio"]["FAD_PANNs"],
            "composite_authenticity": bench["planaudio_best_subjective_authenticity"],
            "sound_clap_planaudio": bench["table4_sound"]["PlanAudio"]["CLAP"],
            "speech_wer_planaudio": bench["table5_speech"]["PlanAudio"]["WER"],
            "best_cot_scf": bench["table6_cot"]["PlanAudio"]["SCF"],
        },
        "integration": {
            "env": "GOPEX_PLANAUDIO=1",
            "fold_hook": "GOPEX_AV_FOLD_HOOKS=...,planaudio",
            "cli": "./scripts/gopex-planaudio.sh",
        },
    }


def evaluation_demo(cfg: PlanAudioConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PlanAudioConfig()
    rng = np.random.default_rng(7)
    h = rng.standard_normal((cfg.af3_embedding_steps, 16))
    z = downsample_af3_embeddings(h, cfg=cfg)
    z_proj = z.mean(axis=0)
    h_tgt = h.mean(axis=0)
    latent = latent_supervision_loss(z_proj, h_tgt, cfg=cfg)
    tot = total_loss(latent["L_latent"], 2.5, cfg=cfg)
    seq = format_sequence(["Upbeat", "music", "starts"], cfg=cfg)
    sample_prompt = (
        "Upbeat music starts playing, Trevor Noah says 'This is great!', followed by loud applause"
    )
    comp = score_prompt_composition(sample_prompt)
    bench = benchmarks_bundle()
    norm = normalized_scenario_score(
        bench["table2_composite"]["PlanAudio"],
        lower_keys=("FAD_PANNs", "FAD_PaSST", "KL_PaSST", "KL_PANNs", "WER"),
        higher_keys=("IS", "CLAP", "UTMOS"),
        reference={
            "FAD_PANNs": {"best": 0.0, "worst": 30.0},
            "CLAP": {"best": 0.25, "worst": 0.10},
            "WER": {"best": 0.05, "worst": 0.8},
        },
    )
    return {
        "framework": framework_card(cfg),
        "cot_smoke": {
            "latent_loss": latent,
            "L_total": tot,
            "sequence_layout": seq,
            "z_shape": list(z.shape),
        },
        "prompt_smoke": {
            "text": sample_prompt,
            "classification": comp,
            "expected_scenario": SCENARIO_COMPOSITE,
        },
        "normalized_composite_score_stub": round(norm, 4),
        "paper_tables": bench,
        "insights": {
            "latent_cot": "Semantic latent CoT beats explicit/acoustic CoT on SCF (Table 6).",
            "curriculum": "Constant 1/3·1/3·1/3 sampling beats disjoint sequential (Fig. 4).",
            "ltx_relevance": (
                "Use free-form AV captions with quoted speech + ordered SFX for LTX "
                "multi-talker clips; complements MTAVG-Bench 2.0 diagnosis."
            ),
        },
    }


def evaluation_smoke(cfg: PlanAudioConfig | None = None) -> dict[str, Any]:
    ev = evaluation_demo(cfg)
    return {
        "ok": ev["prompt_smoke"]["classification"]["scenario"] == SCENARIO_COMPOSITE,
        "latent_cot_steps": (cfg or PlanAudioConfig()).latent_cot_steps,
        "composite_fad_panns": ev["paper_tables"]["table2_composite"]["PlanAudio"]["FAD_PANNs"],
    }
