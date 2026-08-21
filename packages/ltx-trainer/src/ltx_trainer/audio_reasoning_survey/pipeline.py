"""Framework card + paper excerpt tables for the audio reasoning survey (arXiv:2605.21008)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.audio_reasoning_survey.config import AudioReasoningSurveyConfig
from ltx_trainer.audio_reasoning_survey.layout import LIMITATIONS
from ltx_trainer.audio_reasoning_survey.mock import evaluation_smoke
from ltx_trainer.audio_reasoning_survey.taxonomy import agentic_design_patterns, four_paradigms


def framework_card(cfg: AudioReasoningSurveyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioReasoningSurveyConfig()
    return {
        "name": cfg.title,
        "paper": cfg.paper_arxiv,
        "authors": "Zhihan Guo, Wenqian Cui, Guan-Ting Lin, Daxin Tan, Jingyao Li, Qiyong Zheng, "
        "Dingdong Wang, Jing Xiong, Han Shi, Jiaya Jia, Irwin King",
        "contributions": [
            "Unified formulation: direct P(O|C) vs reasoning-augmented P(R,O|C)=P(R|C)P(O|C,R).",
            "Foundations: LALMs (encoder–projector–LLM), SLMs (speech-out), alignment + post-training.",
            "Taxonomy: Audio-to-Text, Audio-to-Speech (sequential + TWL/TWS), Audio-Visual, Agentic.",
            "Evaluation + open challenges: data reliability, modality hallucination, latency, long-context audio.",
        ],
        "obstacles": (
            "Scarcity of audio-grounded reasoning data",
            "Shortcut learning / modality hallucination",
            "Reasoning depth vs real-time latency in speech interaction",
        ),
        "paradigms": four_paradigms(),
        "agentic_patterns": agentic_design_patterns(),
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }


def table2_rl_reward_types_excerpt() -> list[dict[str, Any]]:
    """Table II excerpt — reward dimensions for selected RL audio-to-text methods."""
    cols = ("accuracy", "consistency", "format", "length", "quality")
    rows: list[dict[str, Any]] = [
        {"model": "Step-Audio 2", "rewards": dict(zip(cols, (True, False, False, True, False)))},
        {"model": "Audio-Thinker", "rewards": dict(zip(cols, (True, True, True, False, True)))},
        {"model": "CESAR", "rewards": dict(zip(cols, (True, True, True, True, False)))},
        {"model": "Omni-R1", "rewards": dict(zip(cols, (True, False, False, False, False)))},
    ]
    return rows


def table3_audio_to_text_data_excerpt() -> list[dict[str, Any]]:
    """Table III excerpt — training data + CoT construction (subset)."""
    return [
        {"model": "Audio-Reasoner", "source": "AudioSet, AudioCaps, +7", "cot": "Gemini (4-part)", "scale": "1.2M SFT"},
        {"model": "Audio Flamingo Sound-CoT", "source": "AudioSkills, Clotho-AQA, +10", "cot": "LLM–ALM BFS/DFS", "scale": "1.24M SFT"},
        {"model": "R1-AQA", "source": "AVQA", "cot": "none (direct)", "scale": "38k GRPO"},
        {"model": "Omni-R1", "source": "AVQA, VGGSound", "cot": "ChatGPT QA", "scale": "38k–54k GRPO"},
        {"model": "AudioMCQ", "source": "7 datasets", "cot": "Qwen3-235B 3-stage", "scale": "571k SFT+GRPO"},
    ]


def table6_slm_benchmarks_excerpt() -> list[dict[str, Any]]:
    """Table VI excerpt — SLM reasoning benchmarks (subset)."""
    return [
        {"benchmark": "MMAU", "domain": "General audio reasoning", "eval": "closed", "size_k": 10, "tasks": 27, "year": 2024},
        {"benchmark": "MMAU-Pro", "domain": "General audio reasoning", "eval": "closed", "size_k": 0.5, "tasks": 49, "year": 2025},
        {"benchmark": "VoiceAgentBench", "domain": "Multi-turn tool use & reasoning", "eval": "closed", "size_k": 6, "tasks": 6, "year": 2025},
        {"benchmark": "WavBench", "domain": "End-to-end interaction", "eval": "closed/open", "size_k": 2, "tasks": 5, "year": 2026},
    ]


def table5_agent_tradeoffs() -> dict[str, Any]:
    """Table V — predefined workflow vs dynamic tool-calling (summary)."""
    return {
        "dimensions": ("control_flow", "extensibility", "interpretability", "latency", "data_training"),
        "predefined_workflow": "fixed stages; high interpretability; good for structured tasks",
        "dynamic_tool_calling": "LLM decides tools at runtime; flexible; higher coordination cost",
    }


def headline_results(cfg: AudioReasoningSurveyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioReasoningSurveyConfig()
    return {
        "survey_focus": "First dedicated survey of audio reasoning (formulation + foundations + four paradigms + eval).",
        "mmau_scale": f"{cfg.mmau_size_k}K samples, {cfg.mmau_tasks} tasks",
        "mmau_pro_scale": f"{cfg.mmau_pro_size_k}K samples, {cfg.mmau_pro_tasks} tasks",
        "paradigms": 4,
    }


def evaluation_demo(cfg: AudioReasoningSurveyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioReasoningSurveyConfig()
    return {"paper": cfg.paper_arxiv, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle(cfg: AudioReasoningSurveyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioReasoningSurveyConfig()
    return {
        "framework": framework_card(cfg),
        "table2_rl_rewards_excerpt": table2_rl_reward_types_excerpt(),
        "table3_audio_to_text_data_excerpt": table3_audio_to_text_data_excerpt(),
        "table5_agent_tradeoffs": table5_agent_tradeoffs(),
        "table6_slm_benchmarks_excerpt": table6_slm_benchmarks_excerpt(),
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }
