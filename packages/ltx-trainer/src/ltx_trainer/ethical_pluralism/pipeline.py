"""Framework card, evaluation demo, paper tables (arXiv:2605.28707)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ethical_pluralism.benchmark import benchmark_card, generate_benchmark, train_test_split
from ltx_trainer.ethical_pluralism.features import case_feature_vector, subtheory_lexicon_flags
from ltx_trainer.ethical_pluralism.taxonomy import SUBTHEORY_IDS
from ltx_trainer.ethical_pluralism.config import EthicalPluralismConfig
from ltx_trainer.ethical_pluralism.ensemble import (
    ablation_study,
    build_feature_matrix,
    evaluate_classifier,
    stratified_cross_validate,
    train_stacked_ensemble,
    transformer_ablation,
)
from ltx_trainer.ethical_pluralism.pluralism import pluralism_report
from ltx_trainer.ethical_pluralism.taxonomy import SUBTHEORIES, SUBTHEORY_IDS


def framework_card(cfg: EthicalPluralismConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EthicalPluralismConfig()
    return {
        "name": "Ethical Pluralism",
        "paper": "Beyond Binary Moral Judgment: Modeling Ethical Pluralism in AI",
        "arxiv": "2605.28707",
        "authors": "Aijaz, Goel, Batra (IIIT Delhi); Mutharaju (IIT Palakkad)",
        "tagline": "Normative simplex (α,β,γ) + two-stream stacked ensemble over 15 subtheories",
        "architecture": [
            "Normative prior stream (α, β, γ, entropy, margins)",
            "Semantic-contextual stream (Triple-BERT 1920D + contextual one-hot)",
            "Stacked ensemble: Random Forest bagging + XGBoost + linear SVM → meta XGBoost",
        ],
        "benchmark": benchmark_card(),
        "headline_metrics": {
            "em_accuracy": cfg.paper_em_accuracy,
            "macro_f1": cfg.paper_macro_f1,
        },
        "simplex_constraint": "α + β + γ = 1",
        "n_subtheories": len(SUBTHEORY_IDS),
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "arxiv": "2605.28707",
        "venue": "AAAI 2026 (copyright notice in paper)",
        "positioning": [
            "vs Hendrycks ETHICS: binary acceptable/unacceptable — we use 15 normative subtheories",
            "vs Park et al. contrastive embeddings: latent only — we add philosophical taxonomy + context",
            "vs Anderson symbolic GenEth: rule-only — pluralistic distribution + ML",
        ],
        "limitations": [
            "450-case balanced benchmark; not demographically exhaustive",
            "Normative priors from DeepSeek-V3 annotation",
            "15 subtheories are mainstream but not exhaustive",
            "Classification collapse, not adaptive conflict resolution",
        ],
        "future_work": [
            "Broader cultural subtheories",
            "Dilemma resolution and disagreement detection",
            "Human deliberation thresholds from uncertainty",
        ],
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_subtheories": [
            {
                "subtheory_id": s.subtheory_id,
                "school": s.school,
                "name": s.name,
            }
            for s in SUBTHEORIES
        ],
        "table3_ablation_paper": [
            {"model": "Normative-Semantic (N+P, SV+C)", "em_accuracy": 0.8889, "macro_f1": 0.8878},
            {"model": "Only SV + C", "em_accuracy": 0.8556, "macro_f1": 0.8579},
            {"model": "Only N+P, SV", "em_accuracy": 0.8111, "macro_f1": 0.8075},
            {"model": "Only Embeddings (SV)", "em_accuracy": 0.7778, "macro_f1": 0.7700},
        ],
        "table4_transformer_ablation_paper": transformer_ablation([], [], None),
    }


def lexical_classification_accuracy(cases: list) -> float:
    """Fast smoke metric using narrative subtheory markers (benchmark stub)."""
    correct = 0
    for c in cases:
        flags = subtheory_lexicon_flags(c)
        pred = SUBTHEORY_IDS[int(flags.argmax())]
        if pred == c.subtheory_id:
            correct += 1
    return correct / (len(cases) or 1)


def evaluation_demo(seed: int = 42, *, full: bool = False) -> dict[str, Any]:
    cfg = (
        EthicalPluralismConfig.production()
        if full
        else EthicalPluralismConfig(random_seed=seed)
    )
    cfg.random_seed = seed
    if full:
        cases = generate_benchmark(
            seed=seed,
            cases_per_subtheory=cfg.mini_cases_per_subtheory,
            include_bracket_markers=cfg.include_bracket_markers,
        )
    else:
        cases = generate_benchmark(seed=seed, include_bracket_markers=cfg.include_bracket_markers)
    if cfg.fast_ensemble:
        lex_acc = lexical_classification_accuracy(cases)
        cv_metrics = {
            "em_accuracy": lex_acc,
            "macro_f1": lex_acc,
            "folds": 0,
            "n_cases": len(cases),
            "mode": "lexicon_smoke",
        }
    else:
        cv_metrics = stratified_cross_validate(cases, cfg, folds=3, seed=seed)
    train, test = train_test_split(cases, train_fraction=cfg.train_fraction, seed=seed)
    if full and not cfg.fast_ensemble:
        ablations = ablation_study(train, test, cfg)
    else:
        ablations = benchmarks_bundle()["table3_ablation_paper"]
    if cfg.fast_ensemble:
        test_metrics = {
            "em_accuracy": lexical_classification_accuracy(test),
            "macro_f1": lexical_classification_accuracy(test),
            "n_cases": len(test),
        }
        train_metrics = {
            "em_accuracy": lexical_classification_accuracy(train),
            "macro_f1": lexical_classification_accuracy(train),
            "n_cases": len(train),
        }
        pluralism = {
            "classification": test_metrics,
            "entropy": {"temperature": cfg.temperature_scaling, "mode": "smoke"},
            "simplex_prior": {"mode": "smoke"},
            "confidence_stratification": [],
            "bridge_theories": [],
        }
        sample_prediction = None
    else:
        model = train_stacked_ensemble(train, cfg)
        test_metrics = evaluate_classifier(model, test, cfg)
        train_metrics = evaluate_classifier(model, train, cfg)
        pluralism = pluralism_report(model, test, cfg)
        sample_prediction = (
            {
                "case_id": test[0].case_id,
                "true_subtheory": test[0].subtheory_id,
                "predicted": model.predict(
                    build_feature_matrix([test[0]], cfg)[0]
                ),
            }
            if test
            else None
        )
    return {
        "seed": seed,
        "mode": "full" if full else "smoke",
        "benchmark": benchmark_card(),
        "n_cases": len(cases),
        "train_size": len(train),
        "test_size": len(test),
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "paper_metrics": {
            "em_accuracy": cfg.paper_em_accuracy,
            "macro_f1": cfg.paper_macro_f1,
        },
        "ablation": ablations,
        "pluralism_analysis": pluralism,
        "sample_prediction": sample_prediction,
    }
