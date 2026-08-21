"""MARI evaluation smoke (arXiv:2605.28722)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mari.config import MARIConfig
from ltx_trainer.mari.pipeline import evaluation_demo


def evaluation_smoke(cfg: MARIConfig | None = None) -> dict[str, Any]:
    _ = cfg
    demo = evaluation_demo(seed=0)
    return {
        "paper": "arXiv:2605.28722",
        "energy_threshold": demo["energy_threshold"],
        "accuracy_mari": demo["accuracy_mari"],
        "fraction_gated_off": demo["fraction_gated_off"],
        "theory_improves": demo["theory"]["improvement"]["improves_over_single"],
        "paper_mari_llama3_mc1": demo["paper_mari_llama3_mc1"],
    }
