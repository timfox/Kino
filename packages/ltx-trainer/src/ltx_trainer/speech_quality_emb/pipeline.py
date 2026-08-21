"""Framework card and paper benchmark excerpts (arXiv:2605.21332)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.speech_quality_emb.config import SpeechQualityEmbConfig
from ltx_trainer.speech_quality_emb.layout import LIMITATIONS
from ltx_trainer.speech_quality_emb.mock import evaluation_smoke


def framework_card(cfg: SpeechQualityEmbConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpeechQualityEmbConfig()
    return {
        "name": "Speech quality embeddings (LSSQA + partial mix-up + supervised contrastive)",
        "paper": cfg.paper_arxiv,
        "authors": "Michael Kuhlmann, Tobias Cord-Landwehr, Reinhold Haeb-Umbach (Paderborn University)",
        "problem": "Global MOS misses local degradations; need frame-level quality and degradation-type structure.",
        "method": {
            "parallel_corpus": "Clean reference D_ref and degraded D_deg; pretrained LSSQA frame scores Q_ref, Q_deg.",
            "partial_mix_up": "Binary masks m(t), m_q align in time: s_pseudo = m·s_deg + (1−m)·s_ref; q_pseudo = m_q⊙q_deg + (1−m_q)⊙q_ref (Eq. 3–4).",
            "pseudo_supervision": "L^sup_LSSQA = L_LSSQA + (1/BL) Σ |q̂ − q_pseudo| with utterance target y_b = mean(q_pseudo) on mix-up batches (Eq. 5).",
            "contrastive_embeddings": "Second decoder gDec_scl → z_scl → Proj_scl → z̃ on sphere; supervised contrastive L_scl over frames (Eq. 6–7); exclude near-neighbor self-pairs within λ frames.",
            "total_loss": "L_total = L^sup_LSSQA + τ·L_scl (Eq. 8); τ=0.1, λ=10 neighbor exclusion (Sec. 6.1.2).",
            "detection": "MOS-threshold vs embedding cosine-to-clean-enrollment (Sec. 5); metrics: frame-EER, minDCF (p_target=0.01), I-AUC with ρ_DTC=ρ_GTC=0.7.",
        },
        "training_data": "NISQA + BVCC; encoder wav2vec2-large @ 50 Hz, 1024-D; MOS CNN decoder + parallel scl head D_P=128 (Sec. 6.1.2).",
        "open_source": "https://github.com/fgnt/local_sqa (paper footnote); LibriAugmented recipe https://github.com/fgnt/frame-level-mos",
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }


def table1_training_setup() -> list[dict[str, Any]]:
    """Table 1 — training setup (loss, mix-up probability, clean handling)."""
    return [
        {"model_id": "Baseline [3]", "loss": "L_LSSQA (2)", "pmixup": 0.0, "include_clean_in_scl": None},
        {"model_id": "SUP1", "loss": "L^sup_LSSQA (5)", "pmixup": 0.5, "include_clean_in_scl": None},
        {"model_id": "SUP2", "loss": "L^sup_LSSQA (5)", "pmixup": 1.0, "include_clean_in_scl": None},
        {"model_id": "CON1", "loss": "L_total (8)", "pmixup": 1.0, "include_clean_in_scl": True},
        {"model_id": "CON2", "loss": "L_total (8)", "pmixup": 1.0, "include_clean_in_scl": False},
    ]


def table2_nisqa_test_sim_partial_mixup() -> list[dict[str, Any]]:
    """Table 2 excerpt — NISQA TEST SIM-partial-mixup (embedding vs MOS)."""
    return [
        {
            "model_id": "Baseline [3]",
            "frame_eer_pct": {"x": 19.21, "z_MOS": 17.06},
            "frame_minDCF": {"x": 1.00, "z_MOS": 1.00},
            "embed_I_auc_07_07": {"x": 0.04, "z_MOS": 0.02},
            "mos_I_auc_07_07": 0.01,
        },
        {
            "model_id": "SUP1",
            "frame_eer_pct": {"x": 15.36, "z_MOS": 13.34},
            "frame_minDCF": {"x": 0.99, "z_MOS": 0.87},
            "embed_I_auc_07_07": {"x": 0.47, "z_MOS": 0.48},
            "mos_I_auc_07_07": 0.25,
        },
        {
            "model_id": "SUP2",
            "frame_eer_pct": {"x": 12.71, "z_MOS": 11.01},
            "frame_minDCF": {"x": 1.00, "z_MOS": 1.00},
            "embed_I_auc_07_07": {"x": 0.57, "z_MOS": 0.58},
            "mos_I_auc_07_07": 0.52,
        },
        {
            "model_id": "CON1",
            "frame_eer_pct": {"x": 5.22, "z_scl": 3.87, "z_tilde": 3.93},
            "frame_minDCF": {"x": 0.97, "z_scl": 0.60, "z_tilde": 0.60},
            "embed_I_auc_07_07": {"x": 0.86, "z_scl": 0.91, "z_tilde": 0.91},
            "mos_I_auc_07_07": 0.65,
        },
        {
            "model_id": "CON2",
            "frame_eer_pct": {"x": 9.35, "z_scl": 5.39, "z_tilde": 33.8},
            "frame_minDCF": {"x": 0.99, "z_scl": 0.84, "z_tilde": 1.00},
            "embed_I_auc_07_07": {"x": 0.50, "z_scl": 0.81, "z_tilde": 0.10},
            "mos_I_auc_07_07": 0.67,
        },
    ]


def table3_libri_augmented_test_clean() -> list[dict[str, Any]]:
    """Table 3 excerpt — LibriAugmented/test-clean-partial-mixup."""
    return [
        {
            "model_id": "Baseline [3]",
            "frame_eer_pct": {"x": 15.29, "z_MOS": 16.61},
            "embed_I_auc_07_07": {"x": 0.30, "z_MOS": 0.07},
            "mos_I_auc_07_07": 0.20,
        },
        {
            "model_id": "SUP2",
            "frame_eer_pct": {"x": 11.07, "z_MOS": 10.79},
            "embed_I_auc_07_07": {"x": 0.79, "z_MOS": 0.72},
            "mos_I_auc_07_07": 0.76,
        },
        {
            "model_id": "CON1",
            "frame_eer_pct": {"x": 7.32, "z_scl": 4.88, "z_tilde": 4.64},
            "frame_minDCF": {"x": 1.00, "z_scl": 0.62, "z_tilde": 0.61},
            "embed_I_auc_07_07": {"x": 0.86, "z_scl": 0.92, "z_tilde": 0.92},
            "mos_I_auc_07_07": 0.77,
        },
        {
            "model_id": "CON2",
            "frame_eer_pct": {"x": 9.51, "z_scl": 7.60, "z_tilde": 26.7},
            "embed_I_auc_07_07": {"x": 0.78, "z_scl": 0.80, "z_tilde": 0.05},
            "mos_I_auc_07_07": 0.78,
        },
    ]


def table4_verification_retrieval_nisqa() -> list[dict[str, Any]]:
    """Table 4 excerpt — degradation verification + retrieval (K=36), oracle detections."""
    return [
        {"model_id": "Baseline [3]", "eer_pct": {"x": 42.11, "z_MOS": 43.86}, "acc_pct": {"x": 2.12, "z_MOS": 0.62}},
        {"model_id": "SUP2", "eer_pct": {"x": 36.29, "z_MOS": 44.64}, "acc_pct": {"x": 7.14, "z_MOS": 2.12}},
        {"model_id": "CON1", "eer_pct": {"z_scl": 15.17, "z_tilde": 14.15}, "acc_pct": {"z_scl": 30.30, "z_tilde": 27.41}},
        {"model_id": "CON2", "eer_pct": {"z_scl": 13.56, "z_tilde": 14.25}, "acc_pct": {"z_scl": 26.79, "z_tilde": 25.76}},
    ]


def table5_libri_verification_retrieval() -> dict[str, Any]:
    """Table 5 excerpt — LibriAugmented oracle; single vs all degradations (zscl / z̃)."""
    return {
        "single": [
            {"model_id": "SUP2", "eer_pct": {"z_scl": 33.6}, "acc_pct": {"z_scl": 46.8}},
            {"model_id": "CON1", "eer_pct": {"z_scl": 19.4, "z_tilde": 19.2}, "acc_pct": {"z_scl": 77.5, "z_tilde": 78.9}},
        ],
        "all": [
            {"model_id": "CON1", "eer_pct": {"z_scl": 19.7, "z_tilde": 19.5}, "acc_pct": {"z_scl": 65.0, "z_tilde": 65.6}},
        ],
    }


def table6_joint_clustering() -> list[dict[str, Any]]:
    """Table 6 excerpt — agglomerative clustering (ARI, ARIdist, clean ACC)."""
    return [
        {"model_id": "CON1", "detector": "oracle", "nisqa_ARIdist": 0.60, "libri_ARIdist": 0.28},
        {"model_id": "CON2", "detector": "oracle", "nisqa_ARIdist": 0.73, "libri_ARIdist": 0.27},
        {"model_id": "CON1", "detector": "CON1", "nisqa_ARIdist": 0.68, "nisqa_ARIdist_dist": 0.43, "nisqa_ACC_clean": 0.90},
        {"model_id": "CON2", "detector": "CON2", "nisqa_ARIdist": 0.27, "nisqa_ARIdist_dist": 0.48, "nisqa_ACC_clean": 0.14},
        {"model_id": "CON2", "detector": "CON1", "nisqa_ARIdist": 0.28, "nisqa_ARIdist_dist": 0.56, "nisqa_ACC_clean": 0.65},
    ]


def headline_results(cfg: SpeechQualityEmbConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpeechQualityEmbConfig()
    return {
        "nisqa_test_con1_zscl_embedding_IAUC": cfg.con1_nisqa_zscl_embed_iauc,
        "nisqa_test_baseline_mos_IAUC": cfg.baseline_nisqa_mos_iauc,
        "libri_test_con1_zscl_frame_eer_pct": cfg.con1_libri_zscl_eer_pct,
        "k_nisqa_train_degradation_classes": cfg.k_train_distinct_combos,
    }


def evaluation_demo(cfg: SpeechQualityEmbConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpeechQualityEmbConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle(cfg: SpeechQualityEmbConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpeechQualityEmbConfig()
    return {
        "headlines": headline_results(cfg),
        "table1_training_setup": table1_training_setup(),
        "table2_nisqa_test_sim_partial_mixup": table2_nisqa_test_sim_partial_mixup(),
        "table3_libri_augmented": table3_libri_augmented_test_clean(),
        "table4_verification_nisqa": table4_verification_retrieval_nisqa(),
        "table5_libri_verification": table5_libri_verification_retrieval(),
        "table6_clustering": table6_joint_clustering(),
    }
