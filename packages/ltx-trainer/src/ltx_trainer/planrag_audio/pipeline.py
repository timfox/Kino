"""Framework card and PlanRAG-Audio benchmark excerpts (arXiv:2605.20414)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.planrag_audio.config import PlanRagAudioConfig
from ltx_trainer.planrag_audio.layout import LIMITATIONS
from ltx_trainer.planrag_audio.mock import evaluation_smoke, example_plan_dict


def table1_audio_database_example_rows() -> list[dict[str, Any]]:
    """Table 1 — example time-aligned records in D(a)."""
    return [
        {"stream": "transcript", "start_s": 20.50, "end_s": 22.10, "example": "He talks about it"},
        {"stream": "speaker", "start_s": 20.50, "end_s": 22.10, "example": "SPEAKER_07"},
        {"stream": "emotion", "start_s": 20.50, "end_s": 22.10, "example": "Neutral (0.58), ..."},
        {"stream": "sound_event", "start_s": 22.00, "end_s": 27.00, "example": "Speech (0.87), ..."},
    ]


def table2_task_query_examples() -> list[dict[str, Any]]:
    """Table 2 — representative user-level query templates (excerpt)."""
    return [
        {"task": "QA-1", "example": "Who is Bela and why was no single arm able to knock him down?"},
        {"task": "Summarization", "example": "Provide an abstractive summary of the meeting segment between 0 and 600 seconds."},
        {"task": "Diarization", "example": "Perform speaker diarization between 300 and 600 seconds."},
        {"task": "SED", "example": "Detect occurrences of the following sound event label(s): Flamenco"},
        {"task": "Event ordering", "example": "Determine the order of first occurrence for: (1) Music (2) Bird flight ..."},
        {"task": "Speaker-constrained QA", "example": "Work on the utterance from speaker 439. What does the woman say ...?"},
    ]


def table19_keyword_vs_vector_retrieval() -> list[dict[str, Any]]:
    """Appendix G Table 19 — keyword vs vector retrieval (MCQA-style relative scores)."""
    return [
        {"duration_min": 30, "keyword_search": 67.23, "vector_search": 60.40},
        {"duration_min": 540, "keyword_search": 56.07, "vector_search": 57.39},
    ]


def table3_model_configs() -> list[dict[str, Any]]:
    """Table 3 — backbone modules used in experiments."""
    return [
        {"model": "Qwen", "version": "Qwen3-4B-Instruct-2507", "params": "4B"},
        {"model": "Gemini", "version": "Gemini 2.5 Flash", "params": "undisclosed"},
        {"model": "Voxtral", "version": "Voxtral-Mini-3B-2507", "params": "5B"},
        {"model": "ASR", "version": "OWSM-CTC v4 medium", "params": "1.01B"},
        {"model": "SED", "version": "BEATs iter3+, AS2M finetuned", "params": "90M"},
        {"model": "SD", "version": "Pyannote, community-1", "params": "8.1M"},
        {"model": "ER", "version": "Odyssey 2024 SER baseline", "params": "316M"},
    ]


def table4_mcqa_llm_tokens_60min() -> list[dict[str, Any]]:
    """Table 4 — average LLM input tokens for MCQA at 60 minutes."""
    return [
        {"model": "Gemini", "avg_tokens_k": 115.2},
        {"model": "Gemini + PlanRAG-Audio", "avg_tokens_k": 0.9},
        {"model": "Qwen + PlanRAG-Audio", "avg_tokens_k": 1.2},
    ]


def table5_speaker_count_event_order() -> list[dict[str, Any]]:
    """Table 5 — advanced single-modality reasoning."""
    return [
        {"model": "Voxtral", "speaker_count_acc_pct": 9.17, "event_order_spearman": -0.10},
        {"model": "Gemini", "speaker_count_acc_pct": 14.20, "event_order_spearman": 0.30},
        {"model": "Gemini + PlanRAG-Audio", "speaker_count_acc_pct": 69.40, "event_order_spearman": 0.68},
        {"model": "Qwen", "speaker_count_acc_pct": 35.16, "event_order_spearman": 0.11},
        {"model": "Qwen + PlanRAG-Audio", "speaker_count_acc_pct": 36.66, "event_order_spearman": 0.34},
    ]


def table6_speaker_constrained_mcqa() -> list[dict[str, Any]]:
    """Table 6 — speaker-constrained MCQA (SC = speaker constraint)."""
    return [
        {"model": "Gemini", "sc": False, "qa_acc_pct": 58.83, "abstain_acc_pct": None},
        {"model": "Gemini", "sc": True, "qa_acc_pct": 68.13, "abstain_acc_pct": 0.54},
        {"model": "Gemini + PlanRAG-Audio", "sc": False, "qa_acc_pct": 65.00, "abstain_acc_pct": None},
        {"model": "Gemini + PlanRAG-Audio", "sc": True, "qa_acc_pct": 70.96, "abstain_acc_pct": 94.90},
        {"model": "Qwen + PlanRAG-Audio", "sc": False, "qa_acc_pct": 65.09, "abstain_acc_pct": None},
        {"model": "Qwen + PlanRAG-Audio", "sc": True, "qa_acc_pct": 67.59, "abstain_acc_pct": 82.20},
    ]


def table7_error_decomposition_rows() -> list[dict[str, Any]]:
    """Appendix B Table 7 — topline vs parseable vs e2e (excerpt)."""
    return [
        {"duration_min": 10, "topline": 79.40, "planrag_parseable": 65.67, "planrag_e2e": 50.05},
        {"duration_min": 60, "topline": 78.90, "planrag_parseable": 65.09, "planrag_e2e": 52.25},
        {"duration_min": 540, "topline": 75.56, "planrag_parseable": 56.70, "planrag_e2e": 41.04},
    ]


def framework_card(cfg: PlanRagAudioConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PlanRagAudioConfig()
    return {
        "name": cfg.title,
        "paper": cfg.paper_arxiv,
        "paper_url": cfg.paper_url,
        "authors": "Masao Someki, Chien-yu Huang, Siddhant Arora, Samuele Cornell, Markus Müller, Nathan Susanj, Rupak V Swaminathan, Grant P Strimel, Jing Liu, Shinji Watanabe (CMU LTI, Amazon AGI)",
        "problem": "Long-form audio blows up LALM context; transcript-only RAG drops paralinguistics and long-range cross-modal structure.",
        "method": {
            "stage1": "Build structured time-aligned database D(a): diarization-aligned transcript, speaker, emotion; independent windows for sound_event (§3.2).",
            "stage2": "Planning LLM emits constrained Θ(q): streams, filters, fusion anchor, return fields, answer_schema (§3.1.2).",
            "stage3": "Rule-based SQL: per-stream CTEs, temporal join / nearest-midpoint fusion with τ seconds tolerance (Appendix F).",
            "stage4": "Generation LLM over retrieved segments R(q, a) with planned output schema (§3.1.4).",
        },
        "retrieval": f"Keyword-based retriever in paper; vector ablation Table 19 (Appendix G); fusion τ = {cfg.fusion_tau_seconds} s.",
        "evaluation": f"Base and advanced tasks; long-form lengths {list(cfg.eval_durations_min)} min (§4.1).",
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }


def headline_results(cfg: PlanRagAudioConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PlanRagAudioConfig()
    ratio = cfg.gemini_planrag_tokens_k / cfg.gemini_full_tokens_k
    return {
        "gemini_mcqa_60min_token_reduction_factor": round(1.0 / ratio, 0) if ratio > 0 else None,
        "gemini_plus_planrag_speaker_count_acc_pct": 69.40,
        "gemini_plus_planrag_sc_abstain_acc_pct": 94.90,
    }


def evaluation_demo(cfg: PlanRagAudioConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PlanRagAudioConfig()
    return {"paper": cfg.paper_arxiv, "example_plan": example_plan_dict(), "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle(cfg: PlanRagAudioConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PlanRagAudioConfig()
    return {
        "framework": framework_card(cfg),
        "table1_db_example": table1_audio_database_example_rows(),
        "table2_task_queries": table2_task_query_examples(),
        "table3_models": table3_model_configs(),
        "table19_retrieval_ablation": table19_keyword_vs_vector_retrieval(),
        "table4_mcqa_tokens": table4_mcqa_llm_tokens_60min(),
        "table5_advanced": table5_speaker_count_event_order(),
        "table6_speaker_constrained": table6_speaker_constrained_mcqa(),
        "table7_error_decomposition": table7_error_decomposition_rows(),
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }
