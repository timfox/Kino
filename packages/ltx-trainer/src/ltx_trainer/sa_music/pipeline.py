"""Framework card, benchmark tables, CPU demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sa_music.benchmark import SUBTASKS, accuracy, extract_mcq_answer, total_questions
from ltx_trainer.sa_music.config import SaMusicConfig
from ltx_trainer.sa_music.generation import PROMPT_LEVELS, build_prompt
from ltx_trainer.sa_music.metrics import (
    abc_syntax_rate,
    kl_divergence,
    pearson_r,
    pitch_entropy,
    pitch_histogram,
    repetition_rate,
    scale_adherence_rate,
)


def framework_card(cfg: SaMusicConfig | None = None) -> dict[str, Any]:
    c = cfg or SaMusicConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "benchmark_questions": c.benchmark_questions,
        "models_evaluated": c.models_evaluated,
        "github": c.github_data,
        "headline": headline_results(c),
    }


def headline_results(cfg: SaMusicConfig | None = None) -> dict[str, Any]:
    c = cfg or SaMusicConfig()
    return {
        "gemini_theory_acc": c.gemini_theory_acc,
        "gemini_style_accuracy": c.gemini_style_accuracy,
        "opensource_theory_range": f"{c.opensource_theory_low:.0%}-{c.opensource_theory_high:.0%}",
        "structural_vs_stylistic_gap": c.gemini_genre_accuracy - c.gemini_style_accuracy,
    }


def table_understanding_anchors(cfg: SaMusicConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or SaMusicConfig()
    return [
        {
            "model": "Gemini 2.5 Pro",
            "theory": c.gemini_theory_acc,
            "knowledge": c.gemini_knowledge_acc,
            "continuation": c.gemini_continuation_acc,
        },
        {"model": "GPT-4o", "theory": 0.72, "knowledge": 0.68, "continuation": 0.65},
        {"model": "GPT-3.5", "theory": 0.506, "knowledge": 0.49, "continuation": 0.48},
        {
            "model": "ChatMusician",
            "theory": c.chatmusician_theory,
            "knowledge": c.chatmusician_knowledge,
            "continuation": c.chatmusician_continuation,
        },
        {"model": "Qwen2.5-32B", "theory": 0.362, "knowledge": 0.34, "continuation": 0.693},
    ]


def table2_human_eval(cfg: SaMusicConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or SaMusicConfig()
    return [
        {
            "model": "Gemini 2.5 Pro",
            "structureness": c.gemini_structureness,
            "genre_accuracy": c.gemini_genre_accuracy,
            "style_accuracy": c.gemini_style_accuracy,
        },
        {"model": "GPT-4o", "structureness": 2.96, "genre_accuracy": 0.75, "style_accuracy": 0.40},
        {"model": "Cogito-v1-llama-8B", "structureness": 3.30, "genre_accuracy": 0.65, "style_accuracy": 0.30},
    ]


def table6_automatic_l3(cfg: SaMusicConfig | None = None) -> dict[str, float]:
    c = cfg or SaMusicConfig()
    return {
        "kl_divergence": c.kl_l3_mean,
        "abc_syntax_accuracy": c.abc_syntax_l3_mean,
        "pitch_histogram_entropy": c.pitch_entropy_l3_mean,
        "scale_adherence": 0.923,
        "repetition_rate": 0.196,
    }


def benchmarks_bundle(cfg: SaMusicConfig | None = None) -> dict[str, Any]:
    c = cfg or SaMusicConfig()
    return {
        "subtasks": [{"name": s.name, "questions": s.questions} for s in SUBTASKS],
        "understanding": table_understanding_anchors(c),
        "human_eval": table2_human_eval(c),
        "automatic_l3": table6_automatic_l3(c),
        "telr_levels": [p.level for p in PROMPT_LEVELS],
        "metric_correlations": {
            "kl_style_r": c.kl_style_pearson_r,
            "scale_style_r": c.scale_adherence_style_r,
            "abc_syntax_style_r": c.abc_syntax_style_r,
        },
    }


def pipeline_demo(seed: int = 42, cfg: SaMusicConfig | None = None) -> dict[str, Any]:
    c = cfg or SaMusicConfig()
    ref_abc = "X:1\nK:C\nM:3/4\n|: C D E | F G A |:\n"
    gen_abc = "X:1\nK:C\nM:3/4\n|: C E G | F A G |:\n"

    p_hist = pitch_histogram(["C", "D", "E", "F", "G", "A"])
    q_hist = pitch_histogram(["C", "E", "G", "F", "A", "G"])
    kl = kl_divergence(p_hist, q_hist)
    ent = pitch_entropy(["C", "D", "E", "F", "G", "A", "B"])

    preds = [
        extract_mcq_answer("The correct answer is B"),
        extract_mcq_answer("[[Answer: C]]"),
        extract_mcq_answer("A and B are plausible but D"),
        None,
    ]
    gold = ["B", "C", "D", "A"]
    acc = accuracy(preds, gold)

    l3 = build_prompt(3, lyrics="...", theme="nature", genre="Rabindra")
    understanding = table_understanding_anchors(c)
    gemini = next(r for r in understanding if "Gemini" in r["model"])

    style_scores = [0.40, 0.40, 0.25, 0.30, 0.05]
    kl_scores = [2.1, 2.8, 3.5, 4.0, 5.2]
    r_kl = pearson_r(kl_scores, style_scores)

    return {
        "total_questions": total_questions(),
        "mcq_accuracy_demo": round(acc, 4),
        "kl_divergence": round(kl, 4),
        "pitch_entropy": round(ent, 4),
        "abc_syntax_rate": round(abc_syntax_rate([ref_abc, gen_abc, "invalid"]), 4),
        "scale_adherence_rate": round(scale_adherence_rate([ref_abc, gen_abc]), 4),
        "repetition_rate": round(repetition_rate([ref_abc, gen_abc]), 4),
        "l3_prompt_has_rabindra": "Rabindra" in l3,
        "gemini_beats_opensource": gemini["theory"] > c.opensource_theory_high,
        "style_accuracy_ceiling": c.gemini_style_accuracy,
        "genre_exceeds_style": c.gemini_genre_accuracy > c.gemini_style_accuracy,
        "kl_style_correlation_negative": r_kl < 0,
        "telr_levels": len(PROMPT_LEVELS),
    }


def evaluation_demo(seed: int = 42, cfg: SaMusicConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
