"""Ethical pluralism benchmark configuration (arXiv:2605.28707)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EthicalPluralismConfig:
    """Normative-semantic stream + stacked ensemble parameters."""

    benchmark_cases: int = 450
    cases_per_subtheory: int = 30
    embedding_dim: int = 1920  # Triple-BERT supervector (384 + 768 + 768)
    projection_dim: int = 128  # stub projection for CPU smoke training
    temperature_scaling: float = 0.6  # paper T for entropy normalization
    shapley_samples: int = 100  # unused; reserved for cross-paper symmetry
    train_fraction: float = 0.8
    ridge_lambda: float = 0.1
    ensemble_bootstrap_runs: int = 3
    fast_ensemble: bool = True  # single ridge learner for CPU smoke (paper uses RF+XGB+SVM stack)
    use_lexicon_shortcut: bool = True  # predict from bracket markers when present (smoke only)
    include_bracket_markers: bool = True  # [subtheory_id] tags in benchmark narratives
    include_lexicon_features: bool = True  # 15-dim marker flags in feature vector
    cache_features: bool = True
    mini_cases_per_subtheory: int = 5  # for --full demo / ablation retrain
    random_seed: int = 42

    @classmethod
    def production(cls) -> EthicalPluralismConfig:
        """Full stacked path without lexicon shortcuts (slower, no label leakage)."""
        return cls(
            fast_ensemble=False,
            use_lexicon_shortcut=False,
            include_bracket_markers=False,
            include_lexicon_features=False,
            cache_features=True,
        )
    # Paper-reported headline metrics (held-out test on full architecture)
    paper_em_accuracy: float = 0.8889
    paper_macro_f1: float = 0.8878
    ablation_full: tuple[str, ...] = ("normative_priors", "context", "triple_bert_sv")
    transformer_names: tuple[str, ...] = (
        "all-MiniLM-L6-v2",
        "all-distilRoBERTa-v1",
        "multi-qa-mpnet-base-dot-v1",
    )
