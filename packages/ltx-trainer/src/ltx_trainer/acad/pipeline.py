"""Framework card and paper benchmark excerpts (arXiv:2605.22262)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.acad.config import AcadConfig
from ltx_trainer.acad.layout import LIMITATIONS
from ltx_trainer.acad.mock import evaluation_smoke


def framework_card(cfg: AcadConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AcadConfig()
    return {
        "name": cfg.title,
        "paper": cfg.paper_arxiv,
        "authors": "Diep Luong, Konstantinos Drossos, Mikko Heikkinen, Tuomas Virtanen",
        "acad_definition": {
            "input": "noisy mixture x̃",
            "context": "inferred embedding e (here: acoustic scene class statistics)",
            "output": "estimate x̂ with out-of-context (OC) components removed",
        },
        "architecture": {
            "C": "CRNN context extractor pretrained on clean Mel ASC; outputs e",
            "D": "FiLM-conditioned UNet mask on |X̃|; |X̂| = |X̃| ⊙ D(|X̃|, e)",
            "training": "stage1 L_ASC on clean; stage2 L_den (SI-SNR) with C frozen or finetuned",
        },
        "dataset": {
            "clean": "CochlScene (6 classes: Kitchen, Park, Restaurant, Restroom, Street, Subway)",
            "oc_pool": "FSD50K events; IC/OC sets from PANNs + AudioSet ontology + manual refinement",
            "synthesis": "Scaper; 10 s; N_train=10k/class, N_val=N_test=3k/class",
            "zenodo": cfg.zenodo_dataset,
        },
        "headlines": headline_results(),
        "limitations": LIMITATIONS,
    }


def table1_model_setups() -> list[dict[str, Any]]:
    """Table 1: context utilization."""
    return [
        {"model": "UNet", "context_utilization": "None"},
        {"model": "UNetASC", "context_utilization": "ASC embedding"},
        {"model": "UNetoracle", "context_utilization": "Oracle scene class"},
        {"model": "UNetconst", "context_utilization": "Uninformative constant vector"},
    ]


def table2_metrics() -> list[dict[str, Any]]:
    """Table 2: mean SI-SDR / SDR (dB) on test split."""
    return [
        {"model": "Noisy input", "si_sdr_mean": 4.27, "si_sdr_std": 0.00, "sdr_mean": 4.26, "sdr_std": 0.00},
        {"model": "UNet", "si_sdr_mean": 10.16, "si_sdr_std": 0.02, "sdr_mean": 10.56, "sdr_std": 0.02},
        {"model": "UNetTu-ASC", "si_sdr_mean": 12.12, "si_sdr_std": 0.04, "sdr_mean": 12.56, "sdr_std": 0.04},
        {"model": "UNetFr-ASC", "si_sdr_mean": 11.04, "si_sdr_std": 0.07, "sdr_mean": 11.47, "sdr_std": 0.09},
        {"model": "UNetconst (emb I)", "si_sdr_mean": 10.02, "si_sdr_std": 0.01, "sdr_mean": 10.41, "sdr_std": 0.03},
        {"model": "UNetoracle (emb II)", "si_sdr_mean": 10.82, "si_sdr_std": 0.02, "sdr_mean": 11.23, "sdr_std": 0.03},
        {"model": "UNetconst (emb II)", "si_sdr_mean": 10.13, "si_sdr_std": 0.03, "sdr_mean": 10.53, "sdr_std": 0.05},
    ]


def headline_results() -> dict[str, Any]:
    cfg = AcadConfig()
    gain_si = cfg.unet_tu_asc_si_sdr_db - cfg.unet_si_sdr_db
    return {
        "unet_si_sdr_db": cfg.unet_si_sdr_db,
        "unet_tu_asc_si_sdr_db": cfg.unet_tu_asc_si_sdr_db,
        "si_sdr_gain_db_vs_unet": round(gain_si, 2),
        "asc_extractor_test_accuracy_pct": cfg.asc_test_accuracy_pct,
        "learned_context_beats_oracle_si_sdr": True,
    }


def evaluation_demo(cfg: AcadConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AcadConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_model_setups": table1_model_setups(),
        "table2_metrics": table2_metrics(),
        "headlines": headline_results(),
    }
