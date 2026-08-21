"""Framework card, benchmark tables, and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.voxcpm2.audiovae import audiovae_demo, table10_reconstruction
from ltx_trainer.voxcpm2.backbone import backbone_demo, table1_family_config
from ltx_trainer.voxcpm2.config import Voxcpm2Config
from ltx_trainer.voxcpm2.sequence import sequence_demo, sequence_table


def framework_card(cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    c = cfg or Voxcpm2Config()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "license": c.license,
        "github": c.github,
        "hf_model": c.hf_model,
        "hf_demo": c.hf_demo,
        "backbone": c.backbone,
        "params_b": c.params_b,
        "components": list(c.pipeline_components),
        "capabilities": list(c.generation_modes),
        "languages": c.n_languages,
        "chinese_dialects": c.n_chinese_dialects,
        "training_hours_m": c.training_hours_m,
        "headline": headline_results(c),
    }


def table3_seed_tts_eval() -> list[dict[str, Any]]:
    """Table 3 — selected Seed-TTS-Eval rows + VoxCPM2."""
    return [
        {"model": "Fish Audio S2", "os": True, "en_wer": 0.99, "en_sim": None, "zh_cer": 0.54, "zh_sim": None},
        {"model": "Qwen3-TTS", "os": True, "en_wer": 1.23, "en_sim": 71.7, "zh_cer": 1.22, "zh_sim": 77.0},
        {"model": "LongCat-Audio-DiT", "os": True, "en_wer": 1.50, "en_sim": 78.6, "zh_cer": 1.09, "zh_sim": 81.8},
        {"model": "VoxCPM", "os": True, "en_wer": 1.85, "en_sim": 72.9, "zh_cer": 0.93, "zh_sim": 77.2},
        {"model": "VoxCPM1.5", "os": True, "en_wer": 2.12, "en_sim": 71.4, "zh_cer": 1.18, "zh_sim": 77.0},
        {"model": "VoxCPM2", "os": True, "en_wer": 1.84, "en_sim": 75.3, "zh_cer": 0.97, "zh_sim": 79.5},
    ]


def table4_inference_recipes() -> list[dict[str, Any]]:
    """Table 4 — Seed-TTS-Eval inference recipes."""
    return [
        {
            "recipe": "Continuation only",
            "en_wer": 1.01,
            "en_sim": 77.7,
            "zh_cer": 1.97,
            "zh_sim": 72.6,
        },
        {
            "recipe": "Reference only",
            "en_wer": 1.10,
            "en_sim": 75.3,
            "zh_cer": 1.81,
            "zh_sim": 67.0,
        },
        {
            "recipe": "Reference + Continuation",
            "en_wer": 0.99,
            "en_sim": 79.5,
            "zh_cer": 1.94,
            "zh_sim": 75.2,
        },
    ]


def table7_minimax_sim_summary() -> dict[str, Any]:
    """Table 7 summary — VoxCPM2 leads SIM on 22/24 languages."""
    return {
        "benchmark": "MiniMax-MLS-Test",
        "languages": 24,
        "voxcpm2_best_sim_count": 22,
        "examples": {
            "English": {"VoxCPM2": 85.4, "Fish Audio S2": 79.7},
            "Chinese": {"VoxCPM2": 82.5, "Fish Audio S2": 81.6},
            "Finnish": {"VoxCPM2": 89.0, "Fish Audio S2": 81.9},
        },
    }


def table9_instruct_tts_eval() -> list[dict[str, Any]]:
    """Table 9 — InstructTTSEval EN/ZH anchors."""
    return [
        {"model": "Gemini-TTS-Pro", "aps_en": 87.6, "dsd_en": 86.0, "rp_en": 67.2, "aps_zh": 89.0},
        {"model": "Qwen3-TTS-VD", "aps_en": 82.9, "dsd_en": 82.4, "rp_en": 68.4, "aps_zh": 85.2},
        {"model": "VoxCPM2", "aps_en": 84.2, "dsd_en": 83.2, "rp_en": 71.4, "aps_zh": 85.2},
    ]


def table11_inference_efficiency() -> list[dict[str, Any]]:
    """Table 11 — RTX 4090 RTF."""
    return [
        {"path": "VoxCPM2 (PyTorch)", "params_b": 2.0, "rtf": 0.30, "vram_gb": 8},
        {"path": "VoxCPM2 (Nano-vLLM)", "params_b": 2.0, "rtf": 0.13, "vram_gb": 8},
        {"path": "VoxCPM1.5 (PyTorch)", "params_b": 0.8, "rtf": 0.15, "vram_gb": 6},
        {"path": "VoxCPM (PyTorch)", "params_b": 0.6, "rtf": 0.17, "vram_gb": 5},
    ]


def training_protocol(cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    c = cfg or Voxcpm2Config()
    return {
        "stages": [
            "multilingual_tts_pretraining",
            "joint_tts_and_controllable_pretraining",
            "high_quality_annealing_sft",
        ],
        "cfg_dropout": c.cfg_dropout_train,
        "cfg_alpha_infer": c.cfg_alpha,
        "inference_timesteps": c.inference_timesteps,
        "max_seq_stage12": 4096,
        "max_seq_stage3": 8192,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_family_config": table1_family_config(),
        "table2_sequence_layouts": sequence_table(),
        "table3_seed_tts_eval": table3_seed_tts_eval(),
        "table4_inference_recipes": table4_inference_recipes(),
        "table7_minimax_sim_summary": table7_minimax_sim_summary(),
        "table9_instruct_tts_eval": table9_instruct_tts_eval(),
        "table10_reconstruction": table10_reconstruction(),
        "table11_inference_efficiency": table11_inference_efficiency(),
        "training_protocol": training_protocol(),
    }


def headline_results(cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    c = cfg or Voxcpm2Config()
    return {
        "params_b": c.params_b,
        "internal_30lang_avg_wer": c.internal_30lang_avg_wer,
        "seed_en_wer": c.seed_en_wer,
        "seed_en_sim": c.seed_en_sim,
        "seed_ref_cont_en_sim": c.seed_ref_cont_en_sim,
        "instruct_rp_en": c.instruct_rp_en,
        "rtf_nanovllm": c.rtf_nanovllm,
        "output_hz": c.decode_sample_rate_hz,
    }


def evaluation_demo(*, seed: int = 0, cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    _ = seed
    c = cfg or Voxcpm2Config()
    return {
        "framework": framework_card(c),
        "sequence": sequence_demo(c),
        "audiovae": audiovae_demo(c),
        "backbone": backbone_demo(),
        "headline": headline_results(c),
    }
