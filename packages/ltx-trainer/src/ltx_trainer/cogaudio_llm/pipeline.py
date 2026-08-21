"""Framework card and benchmark tables."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cogaudio_llm.config import CogAudioLlmConfig
from ltx_trainer.cogaudio_llm.dr_sapo import dr_sapo_demo
from ltx_trainer.cogaudio_llm.eips import eips_demo
from ltx_trainer.cogaudio_llm.lime import lime_demo, lime_statistics


def framework_card(cfg: CogAudioLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CogAudioLlmConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "framework": cfg.framework,
        "backbone": cfg.backbone,
        "dataset": cfg.dataset,
        "repo_url": cfg.repo_url,
        "components": [
            "lime_440k_semantic_acoustic_decoupling",
            "eips_4step_cot",
            "three_stage_sft_internalization",
            "dr_sapo_dual_route_rl",
        ],
        "training_stages": [
            "stage1_explicit_eips_sft",
            "stage2_mixed_prompt_internalization",
            "stage3_dr_sapo_alignment",
        ],
        "lime": {
            "utterances": cfg.lime_utterances,
            "hours": cfg.lime_hours,
            "emotions": cfg.lime_emotions,
        },
        "headline": {
            "empathy_humdial_conflict": cfg.empathy_humdial_conflict,
            "emo_acc_conflict": cfg.emo_acc_conflict,
            "human_empathy_humdial": cfg.empathy_human_humdial,
        },
    }


def table2_empathy_quality() -> list[dict[str, Any]]:
    """Table 2 — empathy quality (implicit response, 1–4 scale)."""
    return [
        {
            "model": "Freeze-Omni",
            "llm_esd_conf": 1.34,
            "llm_esd_nonconf": 1.34,
            "llm_humdial_conf": 2.12,
            "llm_humdial_nonconf": 1.56,
            "human_humdial": 1.90,
        },
        {
            "model": "GLM-4-Voice",
            "llm_esd_conf": 1.42,
            "llm_esd_nonconf": 1.79,
            "llm_humdial_conf": 2.09,
            "llm_humdial_nonconf": 1.67,
            "human_humdial": 2.21,
        },
        {
            "model": "Kimi-Audio",
            "llm_esd_conf": 1.54,
            "llm_esd_nonconf": 1.53,
            "llm_humdial_conf": 2.16,
            "llm_humdial_nonconf": 1.90,
            "human_humdial": 2.29,
        },
        {
            "model": "Step-Audio-2-mini",
            "llm_esd_conf": 1.22,
            "llm_esd_nonconf": 1.58,
            "llm_humdial_conf": 1.95,
            "llm_humdial_nonconf": 1.89,
            "human_humdial": 2.11,
        },
        {
            "model": "Qwen2.5-Omni-7B",
            "llm_esd_conf": 1.64,
            "llm_esd_nonconf": 1.75,
            "llm_humdial_conf": 2.40,
            "llm_humdial_nonconf": 2.14,
            "human_humdial": 2.51,
        },
        {
            "model": "Qwen3-Omni-30B",
            "llm_esd_conf": 1.52,
            "llm_esd_nonconf": 1.86,
            "llm_humdial_conf": 2.38,
            "llm_humdial_nonconf": 1.78,
            "human_humdial": 2.01,
        },
        {
            "model": "GPT-4o-Audio",
            "llm_esd_conf": 1.59,
            "llm_esd_nonconf": 1.82,
            "llm_humdial_conf": 2.58,
            "llm_humdial_nonconf": 1.68,
            "human_humdial": 2.05,
        },
        {
            "model": "CogAudio-LLM",
            "llm_esd_conf": 2.90,
            "llm_esd_nonconf": 2.91,
            "llm_humdial_conf": 3.24,
            "llm_humdial_nonconf": 3.16,
            "human_humdial": 3.17,
        },
    ]


def table3_emotion_accuracy() -> list[dict[str, Any]]:
    """Table 3 — emotion perception accuracy (%)."""
    return [
        {
            "model": "Qwen2.5-omni (Base)",
            "overall": 26.5,
            "conflict": 24.0,
            "nonconf": 68.0,
        },
        {"model": "A. Base (Direct SFT)", "overall": None, "conflict": None, "nonconf": None},
        {"model": "B. Explicit Only SFT", "overall": 47.5, "conflict": 42.0, "nonconf": 73.0},
        {"model": "C. Ours w/o RL", "overall": 47.0, "conflict": 44.0, "nonconf": 73.0},
        {"model": "D. Ours (Full) w/ RL", "overall": 49.5, "conflict": 46.0, "nonconf": 71.0},
    ]


def table4_empathy_ablation() -> list[dict[str, Any]]:
    """Table 4 — empathy ablation (1–4 scale)."""
    return [
        {
            "model": "Qwen2.5-omni (Base)",
            "impl_esd_conf": 1.64,
            "impl_esd_nonconf": 1.75,
            "impl_humdial": 2.40,
            "expl_esd_conf": 1.54,
            "expl_esd_nonconf": 1.64,
            "expl_humdial": 2.64,
        },
        {
            "model": "A. Base (Direct SFT)",
            "impl_esd_conf": 2.10,
            "impl_esd_nonconf": 2.62,
            "impl_humdial": 3.11,
            "expl_esd_conf": None,
            "expl_esd_nonconf": None,
            "expl_humdial": None,
        },
        {
            "model": "B. Explicit Only SFT",
            "impl_esd_conf": None,
            "impl_esd_nonconf": None,
            "impl_humdial": None,
            "expl_esd_conf": 2.39,
            "expl_esd_nonconf": 2.35,
            "expl_humdial": 3.16,
        },
        {
            "model": "C. Ours w/o RL",
            "impl_esd_conf": 2.26,
            "impl_esd_nonconf": 2.61,
            "impl_humdial": 3.09,
            "expl_esd_conf": 2.43,
            "expl_esd_nonconf": 2.71,
            "expl_humdial": 3.24,
        },
        {
            "model": "D. Ours (Full) w/ RL",
            "impl_esd_conf": 2.90,
            "impl_esd_nonconf": 2.91,
            "impl_humdial": 3.24,
            "expl_esd_conf": 2.92,
            "expl_esd_nonconf": 2.89,
            "expl_humdial": 3.39,
        },
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_lime_statistics": lime_statistics(),
        "table2_empathy_quality": table2_empathy_quality(),
        "table3_emotion_accuracy": table3_emotion_accuracy(),
        "table4_empathy_ablation": table4_empathy_ablation(),
    }


def headline_results(cfg: CogAudioLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CogAudioLlmConfig()
    ours = next(r for r in table2_empathy_quality() if r["model"] == "CogAudio-LLM")
    gpt = next(r for r in table2_empathy_quality() if "GPT" in r["model"])
    base = next(r for r in table3_emotion_accuracy() if "Base" in r["model"] and "Qwen" in r["model"])
    full = next(r for r in table3_emotion_accuracy() if "Full" in r["model"])
    return {
        "best_empathy_humdial_conflict": ours["llm_humdial_conf"],
        "empathy_gain_vs_gpt4o_humdial": round(ours["llm_humdial_conf"] - gpt["llm_humdial_conf"], 2),
        "conflict_acc_gain_vs_base": round(full["conflict"] - base["conflict"], 1),
        "human_empathy_humdial": cfg.empathy_human_humdial,
    }


def evaluation_demo(*, seed: int = 0, cfg: CogAudioLlmConfig | None = None) -> dict[str, Any]:
    _ = seed
    cfg = cfg or CogAudioLlmConfig()
    return {
        "framework": framework_card(cfg),
        "lime": lime_demo(cfg=cfg),
        "eips": eips_demo(cfg=cfg),
        "dr_sapo": dr_sapo_demo(cfg=cfg),
        "headline": headline_results(cfg),
    }
