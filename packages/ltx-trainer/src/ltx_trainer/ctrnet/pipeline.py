"""Framework card and CHiME-6 benchmark excerpts (arXiv:2605.19695)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ctrnet.config import Chime6Split, CtrnetConfig
from ltx_trainer.ctrnet.layout import LIMITATIONS
from ltx_trainer.ctrnet.mock import evaluation_smoke


def table_hyperparameters() -> list[dict[str, Any]]:
    """Table I — key hyper-parameters (excerpt)."""
    cfg = CtrnetConfig()
    return [
        {"symbol": "I, J", "description": "FCP past/future taps", "value": f"{cfg.fcp_past_taps_i}, {cfg.fcp_future_taps_j}"},
        {"symbol": "ξ", "description": "FCP denominator floor", "value": cfg.fcp_xi},
        {"symbol": "β", "description": "Speaker-activity loss weight", "value": cfg.sa_loss_weight_beta},
        {"symbol": "Δ", "description": "Reverb modeling delay (frames)", "value": cfg.reverb_delay_frames_delta},
        {"symbol": "L, E", "description": "PuLSS pseudo-label taps / max sync delay", "value": f"{cfg.pseudo_label_filter_taps_l}, {cfg.max_sync_delay_frames_e}"},
        {"symbol": "δ", "description": "LCTE weight in PuLSS", "value": cfg.cte_loss_weight_delta},
        {"symbol": "α, θ", "description": "Magnitude compression / overlap sampling", "value": f"{cfg.magnitude_compress_alpha}, {cfg.overlap_sampling_theta}"},
    ]


def table_ii_ctrnet_close_talk() -> list[dict[str, Any]]:
    """Table II — CTRnet on CHiME-6 close-talk mixtures (cpWER %, oracle diarization, default ASR excerpt)."""
    return [
        {"id": "0", "system": "Unprocessed mixture", "test_cpwer": 29.4},
        {"id": "1b", "system": "GSS (8-channel)", "test_cpwer": 28.2},
        {"id": "2", "system": "Supervised CTRnet", "test_cpwer": 37.9},
        {"id": "9", "system": "Semi-supervised CTRnet (best)", "test_cpwer": 21.8},
        {"id": "10b", "system": "Semi-supervised CTRnet + noise modeling", "test_cpwer": 21.9},
    ]


def table_iii_pulss_farfield() -> list[dict[str, Any]]:
    """Table III — PuLSS on CHiME-6 far-field mixtures (cpWER %, oracle diarization excerpt)."""
    return [
        {"id": "0", "system": "Unprocessed mixture", "test_cpwer": 62.6},
        {"id": "1", "system": "GSS (24-channel)", "test_cpwer": 38.5},
        {"id": "2", "system": "Supervised PuLSS", "test_cpwer": 49.0},
        {"id": "7b", "system": "PuLSS + fine-tuned Parakeet", "test_cpwer": 19.5},
    ]


def table_iv_oracle_diarization() -> list[dict[str, Any]]:
    """Table IV — comparison with CHiME-7/8 submissions (cpWER %, fine-tuned Parakeet)."""
    return [
        {"system": "ESPnet baseline", "challenge": "CHiME-7", "test": 35.5},
        {"system": "NVIDIA NeMo", "challenge": "CHiME-7", "test": 25.7},
        {"system": "USTC", "challenge": "CHiME-7", "test": 19.8},
        {"system": "STCON", "challenge": "CHiME-8", "test": 23.0},
        {"system": "GSS (24-channel)", "challenge": "—", "test": 29.7},
        {"system": "PuLSS", "challenge": "—", "test": 19.5},
        {"system": "Close-talk + CTRnet", "challenge": "—", "test": 15.0},
    ]


def table_v_estimated_diarization() -> list[dict[str, Any]]:
    """Table V — tcpWER % with estimated diarization (fine-tuned Parakeet)."""
    return [
        {"system": "USTC", "challenge": "CHiME-7", "test_tcpwer": 44.8},
        {"system": "IACAS-Thinkit", "challenge": "CHiME-7", "test_tcpwer": 33.5},
        {"system": "STCON", "challenge": "CHiME-8", "test_tcpwer": 33.6},
        {"system": "GSS + USTC diar.", "challenge": "—", "test_tcpwer": 33.5},
        {"system": "PuLSS + USTC diar.", "challenge": "—", "test_tcpwer": 28.5},
        {"system": "PuLSS + STCON diar.", "challenge": "—", "test_tcpwer": 31.7},
    ]


def framework_card(cfg: CtrnetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CtrnetConfig()
    split = Chime6Split()
    return {
        "name": cfg.title,
        "paper": f"arXiv:{cfg.paper_id}",
        "paper_url": cfg.paper_url,
        "authors": "Zhong-Qiu Wang (SUSTech), Samuele Cornell (CMU LTI)",
        "demo_url": cfg.demo_url,
        "problem": (
            "Close-talk lapel mixtures are high-SNR for the wearer but contain cross-talk; "
            "far-field arrays lack clean per-speaker targets for supervised separation on real data."
        ),
        "ctrnet": {
            "task": "Cross-talk reduction (CTR) — blind deconvolution on close-talk + far-field pairs",
            "training": "Unsupervised MC loss + weakly-supervised speaker-activity (frame muting + LSA)",
            "extensions": "Semi-supervised (sim+real), noise modeling, reverb dereverb, overlap sampling",
            "architecture": cfg.dnn_variant,
        },
        "pulss": {
            "task": "Pseudo-label based far-field speech separation",
            "pseudo_labels": "FCP from CTRnet close-talk estimates at reference mic (Eq. 27)",
            "training": "LPL + δ·LCTE on real mixtures; oracle diarization masks as input features",
            "inference": "Oracle or estimated speaker-activity timestamps",
        },
        "chime6": {
            "speakers": cfg.n_speakers_c,
            "farfield_mics": cfg.n_farfield_mics_p,
            "train_sessions": split.train_sessions,
            "training_blocks": split.training_blocks,
            "block_s": cfg.block_duration_s,
        },
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }


def headline_results(cfg: CtrnetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CtrnetConfig()
    return {
        "ctrnet_close_talk_test_cpwer_pct": 21.8,
        "pulss_farfield_test_cpwer_oracle_pct": 19.5,
        "paper_gss_cpwer_test_oracle_pct": 29.7,
        "pulss_vs_gss_oracle_gap_pp": 10.2,
        "pulss_vs_ustc_chime7_gap_pp": 0.3,
        "pulss_tcpwer_ustc_diar_test_pct": 28.5,
        "birdnet_style_domain_gap_note": "First neural separation to beat GSS on CHiME-6 real conversational data",
    }


def evaluation_demo(cfg: CtrnetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CtrnetConfig()
    return {"paper": f"arXiv:{cfg.paper_id}", "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle(cfg: CtrnetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CtrnetConfig()
    return {
        "framework": framework_card(cfg),
        "table1_hyperparameters": table_hyperparameters(),
        "table2_ctrnet_close_talk": table_ii_ctrnet_close_talk(),
        "table3_pulss_farfield": table_iii_pulss_farfield(),
        "table4_oracle_diarization": table_iv_oracle_diarization(),
        "table5_estimated_diarization": table_v_estimated_diarization(),
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }
