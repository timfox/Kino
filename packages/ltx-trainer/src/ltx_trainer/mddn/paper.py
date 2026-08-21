"""Paper knowledge export."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mddn.benchmarks import benchmarks_bundle
from ltx_trainer.mddn.config import (
    HIDDEN_DIM,
    NUM_MDDB,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
)
from ltx_trainer.mddn.datasets import datasets_card
from ltx_trainer.mddn.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "authors": [
            "Cuixin Yang",
            "Rongkang Dong",
            "Kin-Man Lam",
            "Yuhang Zhang",
            "Guoping Qiu",
        ],
        "affiliation": "PolyU / Guangzhou University / University of Nottingham",
        "components": [
            "MDDE: DDCA (d=1) + D4C (d=2,3) parallel branches",
            "MFF: latitude-adaptive spatial attention fusion",
            "Low-rank decomposition on D4C / offset nets",
        ],
        "hidden_dim": HIDDEN_DIM,
        "num_mddb": NUM_MDDB,
        "contributions": [
            "Multi-level sampling range for stretched ERP poles",
            "Outperforms OSRT/GDGT-OSR on ODI-SR, SUN360, Flickr360-val",
            "~13M params, comparable Mult-Adds to GDGT-OSR lightweight",
        ],
        "datasets": datasets_card(),
        "benchmarks": benchmarks_bundle(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "mddn", **evaluation_demo_run()}
