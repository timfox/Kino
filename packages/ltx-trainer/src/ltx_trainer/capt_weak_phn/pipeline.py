"""Framework card and fixed paper excerpts (arXiv:2605.23593).

Paper: "A study on weakly-supervised training approaches for phoneme-level pronunciation scoring"
Authors: Jazmín Vidal, Luciana Ferrer
"""

from __future__ import annotations

from typing import Any

from ltx_trainer.capt_weak_phn.config import CaptWeakPhnConfig
from ltx_trainer.capt_weak_phn.layout import LIMITATIONS
from ltx_trainer.capt_weak_phn.mock import evaluation_smoke


def framework_card(cfg: CaptWeakPhnConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CaptWeakPhnConfig()
    return {
        "name": "Weakly-supervised phoneme-level pronunciation scoring (GOPT variant + two-stage finetuning)",
        "paper": cfg.paper_arxiv,
        "dataset": cfg.dataset,
        "task": cfg.task,
        "core_ideas": [
            "Induce phoneme-level scoring heads using only word/utterance labels via pooled phoneme scores",
            "Compare BASE vs MEAN vs ATTN pooling for higher-level prediction",
            "Two-stage scenario: train with utterance labels then fine-tune with small word/phoneme labeled subset",
            "Balanced subset selection by binning utterance-level ground-truth scores",
        ],
        "baselines": [
            "GOP (raw average log phone posterior)",
            "GOP features + SVR (phoneme-specific regressors)",
            "GOPT BASE multi-task transformer",
        ],
        "metrics": ["PCC (Pearson correlation)", "MSE (mean squared error)"],
        "limitations": LIMITATIONS,
    }


def table_1_dev_results() -> list[dict[str, Any]]:
    """Table 1 excerpt (development set). Confidence interval sizes omitted in stub."""
    return [
        # UWP
        {"labels": "UWP", "model": "BASE", "utt_pcc": 0.71, "wrd_pcc": 0.53, "phn_pcc": 0.61, "phn_mse": 0.09},
        {"labels": "UWP", "model": "MEAN", "utt_pcc": 0.66, "wrd_pcc": 0.55, "phn_pcc": 0.58, "phn_mse": 0.09},
        {"labels": "UWP", "model": "ATTN", "utt_pcc": 0.69, "wrd_pcc": 0.58, "phn_pcc": 0.59, "phn_mse": 0.09},
        # P
        {"labels": "P", "model": "BASE", "utt_pcc": None, "wrd_pcc": None, "phn_pcc": 0.61, "phn_mse": 0.09},
        # W
        {"labels": "W", "model": "BASE", "utt_pcc": None, "wrd_pcc": 0.52, "phn_pcc": None, "phn_mse": None},
        {"labels": "W", "model": "MEAN", "utt_pcc": None, "wrd_pcc": 0.56, "phn_pcc": 0.54, "phn_mse": 0.10},
        {"labels": "W", "model": "ATTN", "utt_pcc": None, "wrd_pcc": 0.59, "phn_pcc": 0.56, "phn_mse": 0.10},
        # UW
        {"labels": "UW", "model": "BASE", "utt_pcc": 0.71, "wrd_pcc": 0.51, "phn_pcc": None, "phn_mse": None},
        {"labels": "UW", "model": "MEAN", "utt_pcc": 0.68, "wrd_pcc": 0.54, "phn_pcc": 0.50, "phn_mse": 0.22},
        {"labels": "UW", "model": "ATTN", "utt_pcc": 0.69, "wrd_pcc": 0.54, "phn_pcc": 0.53, "phn_mse": 0.10},
        # U
        {"labels": "U", "model": "BASE", "utt_pcc": 0.71, "wrd_pcc": None, "phn_pcc": None, "phn_mse": None},
        {"labels": "U", "model": "MEAN", "utt_pcc": 0.71, "wrd_pcc": None, "phn_pcc": 0.46, "phn_mse": 0.27},
        {"labels": "U", "model": "ATTN", "utt_pcc": 0.71, "wrd_pcc": None, "phn_pcc": 0.46, "phn_mse": 0.23},
    ]


def figure_3_test_pcc() -> list[dict[str, Any]]:
    """Figure 3 phoneme-level PCC on test set (qualitative ordering excerpt).

    The paper's figure includes confidence intervals; this stub keeps only point values for anchoring tests/tools.
    """
    return [
        {"system": "2S TR W-100", "pcc": 0.24},
        {"system": "2S TR P-100", "pcc": 0.26},
        {"system": "GOP", "pcc": 0.34},
        {"system": "1S-U (ATTN)", "pcc": 0.46},
        {"system": "2S FT W-100", "pcc": 0.43},
        {"system": "2S FT W-500", "pcc": 0.52},
        {"system": "2S FT P-100", "pcc": 0.49},
        {"system": "2S FT P-500", "pcc": 0.57},
        {"system": "SVR", "pcc": 0.60},
        {"system": "1S-P (ATTN)", "pcc": 0.62},
    ]


def headline_results() -> dict[str, Any]:
    cfg = CaptWeakPhnConfig()
    return {
        "gop_phoneme_pcc": cfg.gop_phoneme_pcc,
        "dev_u_attn_phoneme_pcc": cfg.table1_u_attn_phn_pcc,
        "dev_u_attn_phoneme_mse": cfg.table1_u_attn_phn_mse,
        "test_fig3_gop_pcc": cfg.fig3_test_gop_pcc,
    }


def evaluation_demo(cfg: CaptWeakPhnConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CaptWeakPhnConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke()}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_1_dev_results": table_1_dev_results(),
        "figure_3_test_pcc": figure_3_test_pcc(),
        "headlines": headline_results(),
    }
