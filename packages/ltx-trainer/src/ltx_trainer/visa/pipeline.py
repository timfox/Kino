"""Framework card, MMAR tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.visa.config import VisaConfig
from ltx_trainer.visa.features import multimodal_feature_bundle
from ltx_trainer.visa.routing import routing_demo
from ltx_trainer.visa.taxonomy import category_registry, routing_strategy_counts
from ltx_trainer.visa.voting import ensemble_vote_inference, majority_vote


def framework_card(cfg: VisaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VisaConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "challenge": cfg.challenge,
        "benchmark": cfg.benchmark,
        "paradigm": "LALM as a Tool",
        "components": [
            "multi_modal_feature_extraction",
            "lalm_voting_inference",
            "category_aware_routing",
        ],
        "models": {
            "lalm_ensemble": list(cfg.lalm_models),
            "llm_judge": cfg.llm_backbone,
            "vlm": cfg.vlm_backbone,
            "captioner": cfg.captioner,
            "sed": cfg.sed_model,
        },
        "headline": {
            "accuracy_pct": cfg.accuracy,
            "rubrics_pct": cfg.rubrics_score,
            "agent_track_rank": cfg.agent_track_rank,
            "mmar_avg_pct": cfg.mmar_avg_accuracy,
        },
        "fine_categories": cfg.num_fine_categories,
        "routing_strategies": cfg.num_routing_strategies,
    }


def table1_mmar_modality() -> list[dict[str, Any]]:
    """Table 1 — MMAR by modality (selected rows + VISA)."""
    return [
        {"model": "Step-Audio-R1", "track": "Single", "avg": 71.5},
        {"model": "Qwen3-Omni Thinking", "track": "Single", "avg": 69.9},
        {"model": "AudioToolAgent", "track": "Agent", "avg": 68.8},
        {"model": "SAR-LM", "track": "Agent", "avg": 69.3},
        {
            "model": "VISA (ours)",
            "track": "Agent",
            "sound": 71.5,
            "music": 62.6,
            "speech": 84.0,
            "sound_music": 63.6,
            "sound_speech": 86.2,
            "music_speech": 81.7,
            "sound_music_speech": 75.0,
            "avg": 77.4,
        },
    ]


def table2_subcategories() -> list[dict[str, Any]]:
    """Table 2 — 16 sub-categories (Avg column)."""
    return [
        {"model": "Step-Audio-R1", "avg": 66.20},
        {"model": "Qwen3-Omni-Thinking", "avg": 59.15},
        {"model": "VISA (ours)", "avg": 70.42},
        {"model": "w/o fine-grained category", "avg": 59.15},
    ]


def table3_routing_taxonomy() -> list[dict[str, Any]]:
    """Table 3 — routing strategy summary."""
    counts = routing_strategy_counts()
    return [
        {"strategy": k, "num_categories": v}
        for k, v in sorted(counts.items(), key=lambda x: -x[1])
    ]


def table4_leaderboard() -> list[dict[str, Any]]:
    """Table 4 — Agent Track final leaderboard (selected)."""
    return [
        {"team": "Team D (1st Place)", "track": "Agent", "rubrics": 69.83, "acc": 76.90},
        {"team": "VISA (ours)", "track": "Agent", "rubrics": 66.23, "acc": 77.40},
        {"team": "Team E (3rd Place)", "track": "Agent", "rubrics": 66.09, "acc": 75.10},
        {"team": "w/o fine-grained category", "track": "Agent", "rubrics": 62.63, "acc": 73.30},
        {"team": "Step-Audio-R1", "track": "Single", "rubrics": 58.76, "acc": 71.50},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_mmar_modality": table1_mmar_modality(),
        "table2_subcategories": table2_subcategories(),
        "table3_routing_taxonomy": table3_routing_taxonomy(),
        "table4_leaderboard": table4_leaderboard(),
        "category_registry": category_registry(),
    }


def headline_results(cfg: VisaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VisaConfig()
    visa = next(r for r in table1_mmar_modality() if "VISA" in r["model"])
    lb = next(r for r in table4_leaderboard() if r["team"] == "VISA (ours)")
    return {
        "best_listed_accuracy": lb["acc"],
        "rubrics_score": lb["rubrics"],
        "mmar_avg": visa["avg"],
        "agent_track_rank": cfg.agent_track_rank,
    }


def evaluation_demo(*, seed: int = 0, cfg: VisaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VisaConfig()
    rng = np.random.default_rng(seed)
    wave = rng.standard_normal(16_000) * 0.1
    choices = ["A", "B", "C", "D"]
    features = multimodal_feature_bundle(
        wave,
        question="From how many seconds to how many seconds was slow motion applied?",
        choices=choices,
        cfg=cfg,
    )
    vote = ensemble_vote_inference(choices=choices, seed=seed, cfg=cfg)
    route = disagree_then_route_safe(
        category_id="duration_estimation",
        vote=vote,
        choices=choices,
        seed=seed,
    )
    winner, has_majority = majority_vote(["A", "A", "B"])
    return {
        "framework": framework_card(cfg),
        "features": features,
        "voting": vote,
        "routing": route,
        "majority_smoke": {"winner": winner, "has_majority": has_majority},
        "routing_demo": routing_demo(seed=seed),
        "headline": headline_results(cfg),
    }


def disagree_then_route_safe(
    *,
    category_id: str,
    vote: dict[str, Any],
    choices: list[str],
    seed: int,
) -> dict[str, Any]:
    from ltx_trainer.visa.routing import disagree_then_route

    q = vote["qwen3_omni_thinking"]["answer"]
    s = vote["step_audio_r1"]["answer"]
    vlm = choices[(seed + 2) % len(choices)]
    return disagree_then_route(
        category_id=category_id,
        qwen_answer=q,
        step_answer=s,
        vlm_answer=vlm,
        llm_selected=q if seed % 2 == 0 else s,
    )
