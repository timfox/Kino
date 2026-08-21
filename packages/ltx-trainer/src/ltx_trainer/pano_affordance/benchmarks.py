"""Reference metrics from Zhu et al. (arXiv:2603.09760)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pano_affordance.config import (
    CODE_URL,
    NUM_AFFORDANCE_CLASSES,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
)

# Table I(a) — 360-AGD one-shot
TABLE1_360_AGD: list[dict[str, Any]] = [
    {
        "method": "OOAL",
        "easy": {"KLD": 2.868, "SIM": 0.117, "NSS": 1.267},
        "hard": {"KLD": 3.067, "SIM": 0.097, "NSS": 1.484},
    },
    {
        "method": "OS-AGDO",
        "easy": {"KLD": 2.853, "SIM": 0.124, "NSS": 1.299},
        "hard": {"KLD": 2.965, "SIM": 0.115, "NSS": 1.484},
    },
    {
        "method": "Ours",
        "easy": {"KLD": 1.270, "SIM": 0.506, "NSS": 4.490},
        "hard": {"KLD": 1.306, "SIM": 0.474, "NSS": 4.398},
    },
]

# Table I(b) — AGD20K generalization
TABLE1_AGD20K: dict[str, Any] = {
    "Ours": {
        "seen": {"KLD": 0.739, "SIM": 0.616, "NSS": 1.750},
        "unseen": {"KLD": 1.185, "SIM": 0.475, "NSS": 1.419},
    },
}

# Table II — component ablation (Hard)
TABLE2_COMPONENTS: list[dict[str, Any]] = [
    {"LoRA": False, "DASM": False, "OSDH": False, "KLD": 1.475, "SIM": 0.416, "NSS": 4.196},
    {"LoRA": True, "DASM": False, "OSDH": False, "KLD": 1.421, "SIM": 0.429, "NSS": 4.257},
    {"LoRA": True, "DASM": True, "OSDH": False, "KLD": 1.380, "SIM": 0.450, "NSS": 4.317},
    {"LoRA": True, "DASM": False, "OSDH": True, "KLD": 1.359, "SIM": 0.448, "NSS": 4.339},
    {"LoRA": True, "DASM": True, "OSDH": True, "KLD": 1.306, "SIM": 0.474, "NSS": 4.398},
]

# Table III — loss ablation (Hard)
TABLE3_LOSSES: list[dict[str, Any]] = [
    {"LKL": False, "LRTC": False, "LBCE": True, "KLD": 1.596, "SIM": 0.395, "NSS": 3.891},
    {"LKL": True, "LRTC": False, "LBCE": False, "KLD": 1.459, "SIM": 0.442, "NSS": 4.374},
    {"LKL": False, "LRTC": True, "LBCE": False, "KLD": 1.430, "SIM": 0.450, "NSS": 4.041},
    {"LKL": True, "LRTC": False, "LBCE": True, "KLD": 1.331, "SIM": 0.493, "NSS": 4.361},
    {"LKL": True, "LRTC": True, "LBCE": True, "KLD": 1.306, "SIM": 0.474, "NSS": 4.398},
]


def table1_ours_hard() -> dict[str, float]:
    return next(r for r in TABLE1_360_AGD if r["method"] == "Ours")["hard"]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "dataset_360_agd": {
            "affordance_classes": NUM_AFFORDANCE_CLASSES,
            "splits": ["Easy", "Hard"],
            "sources_easy": ["360-Indoor", "Gibson"],
            "sources_hard": ["PanoContext", "Sun360"],
        },
        "table1_360_agd": TABLE1_360_AGD,
        "table1_agd20k": TABLE1_AGD20K,
        "table2_components": TABLE2_COMPONENTS,
        "table3_losses": TABLE3_LOSSES,
    }
