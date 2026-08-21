"""Speech quality embeddings stub (arXiv:2605.21332)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeechQualityEmbConfig:
    paper_arxiv: str = "arXiv:2605.21332"
    title: str = (
        "Speech Quality Embeddings for Improved Detection and Classification of Degradations in Speech Signals"
    )

    # Sec. 4.1 / 6.1.2 — supervised contrastive
    tau_temperature: float = 0.1
    lambda_neighbor_frames: int = 10  # paper: λ = half receptive field (400 ms @ 50 Hz)

    # Sec. 6.1 — mix-up
    pmixup_sup1: float = 0.5
    pmixup_sup2_con: float = 1.0

    # NISQA class counts (Sec. 6.1)
    k_train_distinct_combos: int = 899
    k_val_distinct_combos: int = 371
    k_test_nisqa_partial_mixup: int = 36

    # Table 2 headline — CON1, zscl, embedding I-AUC (NISQA TEST SIM-partial-mixup)
    con1_nisqa_zscl_embed_iauc: float = 0.91
    baseline_nisqa_mos_iauc: float = 0.01
    sup2_nisqa_mos_iauc: float = 0.52

    # Table 3 — CON1 LibriAugmented zscl EER (%)
    con1_libri_zscl_eer_pct: float = 4.88

    # Table 4 — CON1 zscl verification (NISQA TEST, all 36 combos)
    con1_nisqa_zscl_verif_eer_pct: float = 14.15
    con1_nisqa_zscl_retrieval_acc_pct: float = 27.41

    # Table 6 — clustering NISQA (oracle / detected), CON1 oracle ARIdist
    con1_nisqa_oracle_aridist: float = 0.60

    # Fold-in sidecar (proxy from LTX audio latents)
    embedding_dim: int = 64
    degraded_similarity_threshold: float = 0.55
