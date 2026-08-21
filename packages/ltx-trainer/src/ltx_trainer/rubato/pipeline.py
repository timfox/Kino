"""Rubato / InterMo framework card, paper tables, and smoke demos (arXiv:2605.24291)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.rubato.config import RubatoConfig, dialect_registry, training_data_rows
from ltx_trainer.rubato.encoding import inverse_sequence_length_weight, timestamp_smoothing_distribution
from ltx_trainer.rubato.intermo import (
    parse_barline,
    tast_first_bar_example,
    validate_measure_metric_sum,
)
from ltx_trainer.rubato.layout import LIMITATIONS
from ltx_trainer.rubato.mock import toy_measure_invalid_sum, toy_measure_valid


def framework_card(cfg: RubatoConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RubatoConfig()
    dialects = dialect_registry()
    return {
        "name": "Rubato + InterMo",
        "paper": cfg.paper_arxiv,
        "demo": cfg.demo_url,
        "intermo": (
            "Intervals (metric fractions + zero-duration barlines) and moments (onset/offset pitch events, "
            "PR:/PL: staff markers on change; canonical order: offsets before onsets, sorted by pitch)."
        ),
        "architecture": {
            "family": cfg.backbone_family,
            "approx_params_m": cfg.approx_params_m,
            "encoder_hz": cfg.encoder_frame_hz,
            "audio_hz": cfg.audio_sample_rate_hz,
            "inference_window_s": cfg.inference_window_seconds,
            "hop_fraction": cfg.inference_hop_fraction,
        },
        "training": {
            "subword_regularization_alpha": cfg.subword_regularization_alpha,
            "timestamp_bins": cfg.timestamp_bins_per_window,
            "timestamp_bin_ms": cfg.timestamp_bin_ms,
            "label_smoothing": {"p_center": cfg.label_smoothing_p_center, "w_bins": cfg.label_smoothing_window_bins},
            "loss_length_norm": "1/|T| cross-entropy weighting (multitask)",
        },
        "rtfx_l40s_beam4": {"AMT": cfg.rtfx_amt, "TAST": cfg.rtfx_tast, "DBD": cfg.rtfx_dbd},
        "vocab": {"total": cfg.vocab_total, "semantic_interval_pieces_approx": cfg.vocab_semantic_approx},
        "dialects": [
            {
                "name": d.name,
                "prompts": list(d.prompt_tokens),
                "description": d.description,
                "inference": d.inference,
            }
            for d in dialects
        ],
        "export": {"formats": list(cfg.export_formats), "renderer": cfg.renderer},
    }


def table_omr_ned() -> dict[str, dict[str, float | int | None]]:
    """Table 2 excerpt — OMR-NED (%, lower better); nfail where given in paper."""
    return {
        "Rubato (TAST)": {"ATEPP": 75.9, "ASAP": 64.3, "ASAP_Beyer": 78.7, "nfail_ATEPP": None, "nfail_ASAP": None, "nfail_Beyer": None},
        "Beat-This → Piano-A2S": {"ATEPP": 88.9, "ASAP": 86.6, "ASAP_Beyer": 89.7, "nfail_ATEPP": None, "nfail_ASAP": None, "nfail_Beyer": None},
        "Tkun → M2ST": {"ATEPP": 85.2, "ASAP": 69.1, "ASAP_Beyer": 89.3, "nfail_ATEPP": 1, "nfail_ASAP": None, "nfail_Beyer": None},
        "Tkun → PM2S": {"ATEPP": 92.0, "ASAP": 89.5, "ASAP_Beyer": 92.7, "nfail_ATEPP": 8, "nfail_ASAP": 2, "nfail_Beyer": 2},
        "Gemini 3.1 Pro": {"ATEPP": None, "ASAP": 98.6, "ASAP_Beyer": 98.9, "nfail_ATEPP": None, "nfail_ASAP": 12, "nfail_Beyer": 2},
    }


def table_temporal_f1() -> dict[str, dict[str, float | None]]:
    """Table 3 excerpt — F1 (%, higher better)."""
    return {
        "Rubato (TAST)": {"F1_downbeat_ASAP": 67.8, "F1_beat_ASAP": 75.8, "F1_note_ASAP": 91.0, "F1_note_MAESTRO": 87.1},
        "Rubato (DBD)": {"F1_downbeat_ASAP": 65.2, "F1_beat_ASAP": 82.6, "F1_note_ASAP": None, "F1_note_MAESTRO": None},
        "Rubato (AMT)": {"F1_downbeat_ASAP": None, "F1_beat_ASAP": None, "F1_note_ASAP": 97.3, "F1_note_MAESTRO": 97.0},
        "Beat-This": {"F1_downbeat_ASAP": 64.9, "F1_beat_ASAP": 79.9, "F1_note_ASAP": None, "F1_note_MAESTRO": None},
        "Tkun": {"F1_downbeat_ASAP": None, "F1_beat_ASAP": None, "F1_note_ASAP": None, "F1_note_MAESTRO": 98.3},
    }


def table_version_matching() -> dict[str, dict[str, float]]:
    """Table 4 excerpt — MAP on ATEPP (%, higher better)."""
    return {
        "Rubato (TAST)": {"MAP_work": 97.4, "MAP_performer": 59.3},
        "Rubato + relative timestamps": {"MAP_work": 96.6, "MAP_performer": 72.3},
        "Beat-This → Piano-A2S": {"MAP_work": 85.7, "MAP_performer": 48.5},
        "Tkun → M2ST": {"MAP_work": 96.1, "MAP_performer": 48.5},
        "CLEWS": {"MAP_work": 97.6, "MAP_performer": 67.8},
    }


def intermo_validation_demo() -> dict[str, Any]:
    ok_bar, ok_iv = toy_measure_valid()
    bad_bar, bad_iv = toy_measure_invalid_sum()
    ex = tast_first_bar_example()
    return {
        "barline_parse": parse_barline("|3/4k-4"),
        "measure_sum_valid": validate_measure_metric_sum(ok_bar, ok_iv),
        "measure_sum_invalid": validate_measure_metric_sum(bad_bar, bad_iv),
        "tast_fragment_chars": len(ex),
        "tast_prefix": ex[:80],
    }


def encoding_demo(cfg: RubatoConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RubatoConfig()
    dist = timestamp_smoothing_distribution(100, cfg=cfg)
    s = sum(dist.values())
    w_amt = inverse_sequence_length_weight(5000)
    w_dbd = inverse_sequence_length_weight(50)
    return {
        "timestamp_smoothing_sum": round(s, 6),
        "timestamp_smoothing_peak_bin": max(dist, key=dist.get),
        "inverse_len_weight_ratio_amt_over_dbd": round(w_amt / w_dbd, 6) if w_dbd else None,
    }


def evaluation_demo(cfg: RubatoConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RubatoConfig()
    return {
        "framework": framework_card(cfg),
        "intermo": intermo_validation_demo(),
        "encoding": encoding_demo(cfg),
        "training_mix": [r.__dict__ for r in training_data_rows()],
        "limitations": list(LIMITATIONS),
        "paper_tables": {
            "omr_ned": table_omr_ned(),
            "temporal_f1": table_temporal_f1(),
            "version_matching": table_version_matching(),
        },
    }
