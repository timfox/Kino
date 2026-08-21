"""Stance simulation audit framework card, tables, and demos (arXiv:2606.06443)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.stance_sim.config import StanceSimConfig
from ltx_trainer.stance_sim.layout import LIMITATIONS
from ltx_trainer.stance_sim.metrics import pct_change_from_baseline
from ltx_trainer.stance_sim.mock import evaluation_smoke, run_revision_audit, sample_instances
from ltx_trainer.stance_sim.revision import RevisionStrategy


def framework_card(cfg: StanceSimConfig | None = None) -> dict[str, Any]:
    cfg = cfg or StanceSimConfig()
    return {
        "name": "Revising Context, Shifting Simulated Stance",
        "paper": cfg.paper_arxiv,
        "authors": "Zhang, Shan, Lyu, Wei, Luo (Fudan / Rochester)",
        "task": "Counterfactual context revision audit for LLM-based stance simulation",
        "idea": (
            "Given Reddit threads on DeepSeek/Claude/Llama: infer target-user stance, "
            "revise last other-user message (paraphrase/explain/add/meme), re-simulate stance, "
            "measure directional shift Δ and transition rates."
        ),
        "stages": [
            "Stage 1: observed vs inferred stance validation (mask last target message)",
            "Stage 2: counterfactual context revision",
            "Stage 3: stance simulation under revised contexts",
            "Stage 4: Δ and stance transition rate evaluation",
        ],
        "dataset": {
            "conversations": cfg.dataset_conversations,
            "subreddits": cfg.dataset_subreddits,
            "posts": cfg.dataset_posts,
            "target_users": cfg.dataset_target_users,
            "by_target": cfg.counts_by_target,
        },
        "revision_strategies": {
            "text": list(cfg.text_revision_strategies),
            "multimodal": list(cfg.multimodal_strategies),
            "meme_ablations": list(cfg.meme_ablations),
        },
        "models": {
            "stance": list(cfg.stance_models),
            "revision": list(cfg.revision_models),
            "meme_image": cfg.meme_image_model,
        },
        "metrics": [
            "accuracy / macro-F1 / weighted-F1 (Stage 1)",
            "average directional stance shift Δ",
            "supportive vs backfire transition rates",
            "LIWC Tone depolarization (add vs meme)",
        ],
        "defaults": cfg.__dict__,
    }


def table_i_baseline_stance() -> list[dict[str, float | str]]:
    """Table 1 — original stance simulation (GPT-5.2 inferred vs observed)."""
    return [
        {"target": "DeepSeek", "accuracy": 77.38, "macro_f1": 77.93, "weighted_f1": 77.39},
        {"target": "Claude", "accuracy": 78.58, "macro_f1": 78.25, "weighted_f1": 78.87},
        {"target": "Llama", "accuracy": 77.02, "macro_f1": 76.05, "weighted_f1": 76.81},
        {"target": "Overall", "accuracy": 77.64, "macro_f1": 78.10, "weighted_f1": 77.81},
    ]


def table_ii_directional_shift(*, stance_model: str = "GPT-5.2") -> list[dict[str, float | str]]:
    """Table 2 / Table 6 excerpt — average directional stance shift (Gemini revision)."""
    rows = [
        {
            "target": "Claude",
            "inferred_mean": 0.234,
            "paraphrase_delta": 0.002,
            "explain_delta": 0.056,
            "add_delta": 0.113,
            "meme_delta": 0.141,
        },
        {
            "target": "DeepSeek",
            "inferred_mean": 0.170,
            "paraphrase_delta": -0.011,
            "explain_delta": -0.008,
            "add_delta": 0.029,
            "meme_delta": 0.076,
        },
        {
            "target": "Llama",
            "inferred_mean": 0.133,
            "paraphrase_delta": -0.010,
            "explain_delta": 0.067,
            "add_delta": 0.125,
            "meme_delta": 0.051,
        },
    ]
    out: list[dict[str, float | str]] = []
    for r in rows:
        base = float(r["inferred_mean"])
        out.append(
            {
                "stance_model": stance_model,
                "target": r["target"],
                "inferred_mean": base,
                "paraphrase_delta": r["paraphrase_delta"],
                "paraphrase_pct": round(pct_change_from_baseline(base, float(r["paraphrase_delta"])), 1),
                "explain_delta": r["explain_delta"],
                "explain_pct": round(pct_change_from_baseline(base, float(r["explain_delta"])), 1),
                "add_delta": r["add_delta"],
                "add_pct": round(pct_change_from_baseline(base, float(r["add_delta"])), 1),
                "meme_delta": r["meme_delta"],
                "meme_pct": round(pct_change_from_baseline(base, float(r["meme_delta"])), 1),
            }
        )
    avg_base = sum(float(r["inferred_mean"]) for r in rows) / len(rows)
    avg_par = sum(float(r["paraphrase_delta"]) for r in rows) / len(rows)
    avg_exp = sum(float(r["explain_delta"]) for r in rows) / len(rows)
    avg_add = sum(float(r["add_delta"]) for r in rows) / len(rows)
    avg_meme = sum(float(r["meme_delta"]) for r in rows) / len(rows)
    out.append(
        {
            "stance_model": stance_model,
            "target": "Average",
            "inferred_mean": round(avg_base, 3),
            "paraphrase_delta": round(avg_par, 3),
            "paraphrase_pct": round(pct_change_from_baseline(avg_base, avg_par), 1),
            "explain_delta": round(avg_exp, 3),
            "explain_pct": round(pct_change_from_baseline(avg_base, avg_exp), 1),
            "add_delta": round(avg_add, 3),
            "add_pct": round(pct_change_from_baseline(avg_base, avg_add), 1),
            "meme_delta": round(avg_meme, 3),
            "meme_pct": round(pct_change_from_baseline(avg_base, avg_meme), 1),
        }
    )
    return out


def table_iii_meme_ablation_methods() -> list[dict[str, str]]:
    """Table 3 — meme ablation variant definitions."""
    return [
        {
            "method": "r_meme",
            "revision_input": "Meme template",
            "inference_input": "Meme image + text",
            "purpose": "Full method",
        },
        {
            "method": "r_white_meme",
            "revision_input": "Meme template",
            "inference_input": "White background + text",
            "purpose": "Test visual information during inference",
        },
        {
            "method": "r_humor",
            "revision_input": "Humor instruction",
            "inference_input": "Meme image + text",
            "purpose": "Replace template with humor guidance",
        },
        {
            "method": "r_caption_cut",
            "revision_input": "Meme caption",
            "inference_input": "Meme image + text",
            "purpose": "Replace template with visual description",
        },
        {
            "method": "r_caption",
            "revision_input": "Meme caption + usage knowledge",
            "inference_input": "Meme image + text",
            "purpose": "Textualized meme knowledge",
        },
    ]


def table_iv_meme_transition_rates() -> list[dict[str, float | str]]:
    """Table 4 — combined neutral+positive transition rates for meme variants."""
    return [
        {"strategy": "r_meme", "Claude": 21.9, "DeepSeek": 26.7, "Llama": 20.6, "average": 23.1},
        {"strategy": "r_white_meme", "Claude": 18.1, "DeepSeek": 22.2, "Llama": 24.5, "average": 21.6},
        {"strategy": "r_humor", "Claude": 12.9, "DeepSeek": 18.5, "Llama": 25.8, "average": 19.1},
        {"strategy": "r_caption_cut", "Claude": 18.1, "DeepSeek": 21.5, "Llama": 20.6, "average": 20.1},
        {"strategy": "r_caption", "Claude": 16.8, "DeepSeek": 22.2, "Llama": 21.9, "average": 20.3},
    ]


def table_v_tone_depolarization() -> list[dict[str, float | str]]:
    """Table 5 — LIWC Tone depolarization rates (add vs meme)."""
    return [
        {
            "target": "DeepSeek",
            "revision": "add",
            "low_to_high_pct": 83.7,
            "high_to_low_pct": 72.8,
            "overall_depolarize_pct": 78.7,
        },
        {
            "target": "DeepSeek",
            "revision": "meme",
            "low_to_high_pct": 47.8,
            "high_to_low_pct": 68.8,
            "overall_depolarize_pct": 56.6,
        },
        {
            "target": "Claude",
            "revision": "add",
            "low_to_high_pct": 81.0,
            "high_to_low_pct": 82.3,
            "overall_depolarize_pct": 81.7,
        },
        {
            "target": "Claude",
            "revision": "meme",
            "low_to_high_pct": 52.0,
            "high_to_low_pct": 66.4,
            "overall_depolarize_pct": 58.6,
        },
        {
            "target": "Llama",
            "revision": "add",
            "low_to_high_pct": 81.2,
            "high_to_low_pct": 77.7,
            "overall_depolarize_pct": 79.4,
        },
        {
            "target": "Llama",
            "revision": "meme",
            "low_to_high_pct": 47.2,
            "high_to_low_pct": 62.8,
            "overall_depolarize_pct": 54.5,
        },
    ]


def table_vi_full_stance_models() -> list[dict[str, float | str]]:
    """Table 6 — directional shift across stance models (Average row per model)."""
    models = [
        ("GPT-5.2", -0.007, 0.031, 0.080, 0.088),
        ("Sonnet-4.6", -0.004, 0.008, 0.094, 0.092),
        ("Qwen3.5-Plus", -0.002, 0.088, 0.215, 0.229),
    ]
    return [
        {
            "stance_model": name,
            "paraphrase_delta_avg": par,
            "explain_delta_avg": exp,
            "add_delta_avg": add,
            "meme_delta_avg": meme,
        }
        for name, par, exp, add, meme in models
    ]


def table_vii_meme_template_sensitivity() -> list[dict[str, float | str]]:
    """Table 7 — average directional shift across five ImgFlip meme templates."""
    return [
        {
            "target": "Claude",
            "inferred_mean": 0.2358,
            "meme_1_delta": 0.1358,
            "meme_2_delta": 0.1547,
            "meme_3_delta": 0.1283,
            "meme_4_delta": 0.1472,
            "meme_5_delta": 0.1302,
        },
        {
            "target": "DeepSeek",
            "inferred_mean": 0.1705,
            "meme_1_delta": 0.0891,
            "meme_2_delta": 0.0649,
            "meme_3_delta": 0.0509,
            "meme_4_delta": 0.0840,
            "meme_5_delta": 0.0891,
        },
        {
            "target": "Llama",
            "inferred_mean": 0.1301,
            "meme_1_delta": 0.0650,
            "meme_2_delta": 0.0569,
            "meme_3_delta": 0.0407,
            "meme_4_delta": 0.0691,
            "meme_5_delta": 0.0285,
        },
    ]


def table_viii_prompt_temperature_agreement() -> dict[str, list[dict[str, float | str]]]:
    """Tables 8–9 — revision prompt/temperature stance agreement (Gemini / Claude)."""
    return {
        "gemini_revision": [
            {"strategy": "add", "temp_0.5": 87.42, "temp_1.0": 86.33, "prompt_2": 87.15, "prompt_3": 87.42},
            {"strategy": "explain", "temp_0.5": 88.69, "temp_1.0": 88.25, "prompt_2": 88.47, "prompt_3": 89.18},
            {"strategy": "paraphrase", "temp_0.5": 92.97, "temp_1.0": 91.76, "prompt_2": 92.31, "prompt_3": 92.92},
        ],
        "claude_revision": [
            {"strategy": "add", "temp_0.5": 90.72, "temp_1.0": 89.71, "prompt_2": 90.02, "prompt_3": 90.57},
            {"strategy": "explain", "temp_0.5": 91.27, "temp_1.0": 90.41, "prompt_2": 90.88, "prompt_3": 90.41},
            {"strategy": "paraphrase", "temp_0.5": 92.28, "temp_1.0": 92.21, "prompt_2": 92.13, "prompt_3": 92.44},
        ],
    }


def figure3_transition_summary() -> dict[str, float]:
    """Fig. 3 headline transition rates (GPT-5.2, all targets)."""
    return {
        "add_negative_reduction_rate_pct": 4.7,
        "meme_neutral_to_supportive_rate_pct": 17.6,
        "meme_backfire_rate_note": "higher than add; interpret with robustness",
    }


def pipeline_demo(cfg: StanceSimConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or StanceSimConfig()
    instances = sample_instances(seed=seed, n=16)
    audits = {
        s.value: run_revision_audit(instances, s, seed=seed + hash(s.value) % 1000)
        for s in (
            RevisionStrategy.PARAPHRASE,
            RevisionStrategy.EXPLAIN,
            RevisionStrategy.ADD,
            RevisionStrategy.MEME,
        )
    }
    return {
        "instances": len(instances),
        "revision_audits": audits,
        "paper_add_avg_pct": 44.8,
        "paper_meme_avg_pct": 49.3,
    }


def evaluation_demo(cfg: StanceSimConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    smoke = evaluation_smoke(seed=0)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["stage1_overall_macro_f1"] = next(
        r["macro_f1"] for r in table_i_baseline_stance() if r["target"] == "Overall"
    )
    demo["evaluation_smoke"] = smoke
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_baseline_stance": table_i_baseline_stance(),
        "table_ii_directional_shift": table_ii_directional_shift(),
        "table_iii_meme_ablation_methods": table_iii_meme_ablation_methods(),
        "table_iv_meme_transition_rates": table_iv_meme_transition_rates(),
        "table_v_tone_depolarization": table_v_tone_depolarization(),
        "table_vi_full_stance_models": table_vi_full_stance_models(),
        "table_vii_meme_template_sensitivity": table_vii_meme_template_sensitivity(),
        "table_viii_prompt_temperature_agreement": table_viii_prompt_temperature_agreement(),
        "figure3_transition_summary": figure3_transition_summary(),
    }
