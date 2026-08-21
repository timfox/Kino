"""Depression benchmark audit framework card and tables."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dep_audit.config import DepAuditConfig
from ltx_trainer.dep_audit.layout import LIMITATIONS
from ltx_trainer.dep_audit.mock import evaluation_smoke
from ltx_trainer.dep_audit.probes import table_i_probes


def framework_card(cfg: DepAuditConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DepAuditConfig()
    return {
        "name": "Multi-Probe Audit of Clinical-Interview Depression Benchmarks",
        "zenodo": cfg.paper_zenodo,
        "code_release": cfg.paper_code_zenodo,
        "authors": "Takehiro Ishikawa, Jon Duke (Georgia Tech)",
        "corpora": ["DAIC", "E-DAIC", "CMDC", "ANDROIDS", "MODMA", "PDCH"],
        "probes": ["A: LOSO", "B: official-split instability", "C: external validation", "D: SRDS topic stress"],
        "probe_a_model": "T+L hybrid (text + frozen LLM depression_prob); LOSO macro-F1 = 0.723",
        "probe_b_sweep": f"{cfg.config_sweep_count} configurations",
        "primary_metric": "participant-level macro-F1",
        "limitations": LIMITATIONS,
    }


def table_iii_loso() -> list[dict[str, Any]]:
    """Table 3 — E-DAIC LOSO."""
    return [
        {
            "config": "L-only",
            "macro_f1": 0.686,
            "auroc": 0.825,
            "ap": 0.666,
            "confusion": "138/51/28/58",
        },
        {
            "config": "T-only",
            "macro_f1": 0.621,
            "auroc": 0.647,
            "ap": 0.438,
            "confusion": "141/48/43/43",
        },
        {
            "config": "T+L",
            "macro_f1": 0.723,
            "auroc": 0.768,
            "ap": 0.592,
            "confusion": "155/34/32/54",
        },
    ]


def table_iv_official_instability() -> list[dict[str, Any]]:
    """Table 4 — 96-config official split audit."""
    return [
        {"statistic": "Pearson CV vs official test", "value": 0.6373},
        {"statistic": "Spearman CV vs official test", "value": 0.7037},
        {"statistic": "Kendall tau", "value": 0.4884},
        {"statistic": "Best-CV config official rank", "value": 20},
        {"statistic": "Best-test config CV rank", "value": 41},
        {"statistic": "Top-3 overlap", "value": 0},
        {"statistic": "Bootstrap p(rank-1) test-best", "value": 0.323},
        {"statistic": "Bootstrap 95% rank range test-best", "value": "1–19"},
    ]


def table_v_cmdc_external() -> list[dict[str, Any]]:
    """Table 5 — CMDC zero-shot external."""
    return [
        {"dataset": "MODMA", "n": 36, "macro_f1": 0.265, "auroc": 0.672},
        {"dataset": "PDCH >= 8", "n": 62, "macro_f1": 0.127, "auroc": 0.564},
        {"dataset": "PDCH >= 17", "n": 62, "macro_f1": 0.361, "auroc": 0.442},
        {"dataset": "PDCH >= 24", "n": 62, "macro_f1": 0.446, "auroc": 0.422},
        {"dataset": "E-DAIC (supp.)", "n": 275, "macro_f1": 0.238, "auroc": 0.579},
        {"dataset": "ANDROIDS (supp.)", "n": 116, "macro_f1": 0.420, "auroc": 0.468},
    ]


def table_vi_androids_external() -> list[dict[str, Any]]:
    """Table 6 — ANDROIDS external by modality (excerpt)."""
    rows = [
        ("CMDC", "Audio", 0.647, 0.768),
        ("CMDC", "Fusion", 0.400, 0.630),
        ("CMDC", "Text", 0.400, 0.288),
        ("E-DAIC", "Audio", 0.250, 0.518),
        ("MODMA", "Audio", 0.390, 0.482),
        ("PDCH", "Fusion", 0.503, 0.520),
    ]
    return [
        {"target": t, "modality": m, "macro_f1": f1, "auroc": auc}
        for t, m, f1, auc in rows
    ]


def table_vii_topic_stress() -> dict[str, Any]:
    """Table 7 — SRDS heavy-minus-neutral shift."""
    return {
        "audio_shift_mean": -0.004,
        "audio_shift_sd": 0.022,
        "text_shift_mean": 0.422,
        "text_shift_sd": 0.022,
        "text_minus_audio_gap_mean": 0.409,
        "text_minus_audio_gap_sd": 0.046,
        "text_positive_seeds": "5/5",
        "permutation_p": 0.0002,
    }


def headline_results() -> dict[str, Any]:
    loso = table_iii_loso()
    return {
        "finding": (
            "E-DAIC LOSO T+L macro-F1 0.723 is a conservative anchor; official-split "
            "leaderboards are unstable (top-3 CV/test overlap = 0); near-ceiling CMDC/ANDROIDS "
            "scores do not zero-shot transfer; text models jump on SRDS symptom-dense slices."
        ),
        "loso_tl_macro_f1": loso[2]["macro_f1"],
        "official_p_rank1": 0.323,
        "cmdc_in_domain_f1_approx": 0.95,
        "cmdc_modma_macro_f1": 0.265,
    }


def evaluation_demo(cfg: DepAuditConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DepAuditConfig()
    return {"zenodo": cfg.paper_zenodo, "smoke": evaluation_smoke()}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "table_i_probes": table_i_probes(),
        "table_iii_loso": table_iii_loso(),
        "table_iv_official_instability": table_iv_official_instability(),
        "table_v_cmdc_external": table_v_cmdc_external(),
        "table_vi_androids_external": table_vi_androids_external(),
        "table_vii_topic_stress": table_vii_topic_stress(),
        "headlines": headline_results(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
