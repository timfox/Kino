"""Age-aware adapter tuning for child ASR — Li, arXiv:2606.05440."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChildAsrAgeAdapterConfig:
    paper_arxiv: str = "arXiv:2606.05440"
    title: str = "Age-Aware Adapter Tuning for Children's Speech Recognition"
    framework: str = "ChildASR-AgeAdapter"
    author: str = "Jialu Li"
    affiliation: str = "University of Arizona"
    code_url: str = "https://github.com/jialuli3/child_asr_age_adapter.git"

    # Challenge / dataset (§2, Table 1)
    challenge: str = "On Top of Pasketti Children's ASR (Word Track)"
    age_groups: tuple[str, ...] = ("3_4_unknown", "5_7", "8_11", "12_plus")
    max_utterance_seconds: float = 30.0

    # Backbone (§3)
    backbone: str = "nvidia/parakeet-tdt-0.6b-v2"
    backbone_params_m: float = 600.0
    conformer_layers: int = 24
    hidden_dim: int = 1024
    router_feature_layer: int = 4

    # Shared child adapter
    child_bottleneck: int = 128
    child_adapter_params_m: float = 6.3
    child_train_steps: int = 50_000

    # Age-specialized adapters
    age_bottleneck: int = 32
    age_adapter_params_m: float = 6.4
    age_train_steps_per_group: int = 50_000

    # Age router
    router_hidden: int = 128
    router_params_k: float = 531.0

    # Unified FiLM adapter
    film_bottleneck: int = 128
    film_params_m: float = 6.7
    film_train_steps: int = 60_000
    film_age_embed_dim: int = 64
    film_gate_init: float = -3.0
    age_homogeneous_alpha: float = 0.3

    # Table 2 — freeze baseline (no child adapter)
    freeze_wer: float = 23.8
    freeze_macro_wer: float = 37.4

    # Table 2 — shared child adapter baseline
    shared_wer: float = 12.6
    shared_macro_wer: float = 18.4

    # Table 2 — best: child shared + age-specialized (GT routing)
    best_wer: float = 12.3
    best_macro_wer: float = 17.6

    # Table 2 — age-specialized only (GT)
    age_only_wer: float = 12.4
    age_only_macro_wer: float = 17.9

    # Table 2 — predicted routing
    pt_top1_wer: float = 12.4
    pt_top1_macro_wer: float = 17.9
    pt_top2_wer: float = 12.3
    pt_top2_macro_wer: float = 17.8

    # Table 2 — unified FiLM (ground-truth, homogeneous batching)
    film_gt_hom_wer: float = 12.7
    film_gt_hom_macro_wer: float = 18.3

    # Table 2 — stacked adapter control (no age conditioning)
    stacked_wer: float = 13.1
    stacked_macro_wer: float = 19.1

    # Table 2 — FiLM predicted-age homogeneous
    film_pt_hom_wer: float = 12.6
    film_pt_hom_macro_wer: float = 18.1

    # Age router (§4.2, Fig. 2)
    router_accuracy: float = 0.743
    router_macro_f1: float = 0.757

    # Per-group WER for best model (Table 2)
    best_wer_3_4: float = 36.9
    best_wer_5_7: float = 14.7
    best_wer_8_11: float = 8.3
    best_wer_12_plus: float = 4.3
    best_wer_unknown: float = 23.8
