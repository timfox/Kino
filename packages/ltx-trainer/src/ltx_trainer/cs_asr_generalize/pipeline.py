"""Framework card, Table 1, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cs_asr_generalize.config import CsAsrGeneralizeConfig
from ltx_trainer.cs_asr_generalize.domain_gen import DgMethod, dg_objective
from ltx_trainer.cs_asr_generalize.merging import MergeMethod, merge_models
from ltx_trainer.cs_asr_generalize.metrics import average_mer


def framework_card(cfg: CsAsrGeneralizeConfig | None = None) -> dict[str, Any]:
    c = cfg or CsAsrGeneralizeConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "cs_asr_unseen_pair_generalization",
        "backbone": c.backbone,
        "metric": c.metric,
        "languages": list(c.languages),
        "seen_pairs": list(c.seen_pairs),
        "unseen_pairs": list(c.unseen_pairs),
        "methods": ["fine-tuning", "model merging (TA/TIES/DARE)", "domain generalization (Fish/Fishr/GGA-L)"],
        "headline": headline_results(c),
    }


def headline_results(cfg: CsAsrGeneralizeConfig | None = None) -> dict[str, Any]:
    c = cfg or CsAsrGeneralizeConfig()
    return {
        "baseline_unseen_avg_mer": c.baseline_unseen_avg,
        "best_ties3_unseen_avg_mer": c.ties3_unseen_avg,
        "best_fishr_unseen_avg_mer": c.fishr_unseen_avg,
        "ko_ja_eval_utterances": c.ko_ja_eval_utterances,
        "ko_de_eval_utterances": c.ko_de_eval_utterances,
        "possible_language_pairs": c.possible_pairs,
    }


def table1_mer_results(cfg: CsAsrGeneralizeConfig | None = None) -> list[dict[str, Any]]:
    """Table 1 — MER on seen and unseen CS pairs."""
    c = cfg or CsAsrGeneralizeConfig()
    rows: list[dict[str, Any]] = [
        {
            "method": "WHISPER-MEDIUM",
            "ko_en": c.baseline_ko_en,
            "ja_en": c.baseline_ja_en,
            "de_en": c.baseline_de_en,
            "seen_avg": c.baseline_seen_avg,
            "ko_de": c.baseline_ko_de,
            "ko_ja": c.baseline_ko_ja,
            "unseen_avg": c.baseline_unseen_avg,
        },
        {
            "method": "TIES (KO-EN + JA-EN + DE-EN)",
            "ko_en": 0.11,
            "ja_en": 0.20,
            "de_en": 0.11,
            "seen_avg": c.ties3_seen_avg,
            "ko_de": c.ties3_ko_de,
            "ko_ja": c.ties3_ko_ja,
            "unseen_avg": c.ties3_unseen_avg,
        },
        {
            "method": "Fishr (DG)",
            "ko_en": 0.11,
            "ja_en": 0.29,
            "de_en": 0.13,
            "seen_avg": c.fishr_seen_avg,
            "ko_de": c.fishr_ko_de,
            "ko_ja": c.fishr_ko_ja,
            "unseen_avg": c.fishr_unseen_avg,
        },
        {
            "method": "TIES (KO-EN + JA-EN)",
            "ko_en": 0.11,
            "ja_en": 0.20,
            "de_en": 0.12,
            "seen_avg": c.ties_ko_ja_en_seen_avg,
            "ko_de": 0.34,
            "ko_ja": 0.31,
            "unseen_avg": c.ties_ko_ja_en_unseen_avg,
        },
    ]
    return rows


def benchmarks_bundle(cfg: CsAsrGeneralizeConfig | None = None) -> dict[str, Any]:
    c = cfg or CsAsrGeneralizeConfig()
    return {
        "table1_mer": table1_mer_results(c),
        "eval_datasets": {
            "ko_ja": {"n": c.ko_ja_eval_utterances, "hub": c.ko_ja_hub},
            "ko_de": {"n": c.ko_de_eval_utterances},
        },
        "merge_methods": [m.value for m in MergeMethod],
        "dg_methods": [m.value for m in DgMethod],
    }


def pipeline_demo(seed: int = 42, cfg: CsAsrGeneralizeConfig | None = None) -> dict[str, Any]:
    """CPU stub: merge three pair-specific deltas + DG gradient stats."""
    c = cfg or CsAsrGeneralizeConfig()
    rng = np.random.default_rng(seed)
    shape = (8, 16)
    deltas = [rng.standard_normal(shape) * 0.01 for _ in c.seen_pairs]
    domain_grads = [rng.standard_normal(64) for _ in c.seen_pairs]

    ties3 = merge_models(MergeMethod.TIES, deltas)
    ta3 = merge_models(MergeMethod.TASK_ARITHMETIC, deltas)
    dare3 = merge_models(MergeMethod.DARE, deltas, seed=seed)
    fishr_obj = dg_objective(DgMethod.FISHR, domain_grads)

    pair_mers = {
        "KO-EN": c.baseline_ko_en,
        "JA-EN": c.baseline_ja_en,
        "DE-EN": c.baseline_de_en,
    }
    return {
        "ties_merge_norm": float(np.linalg.norm(ties3)),
        "task_arithmetic_norm": float(np.linalg.norm(ta3)),
        "dare_merge_norm": float(np.linalg.norm(dare3)),
        "fishr_variance_penalty": fishr_obj["variance_penalty"],
        "baseline_seen_avg_mer": average_mer(pair_mers, c.seen_pairs),
        "baseline_unseen_avg_mer": c.baseline_unseen_avg,
        "best_ties3_unseen_avg_mer": c.ties3_unseen_avg,
        "improvement_vs_baseline": round(c.baseline_unseen_avg - c.ties3_unseen_avg, 2),
    }


def evaluation_demo(seed: int = 42, cfg: CsAsrGeneralizeConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
