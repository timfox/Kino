"""TaC evaluation smoke (arXiv:2605.28713)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.tac.config import TaCConfig
from ltx_trainer.tac.pipeline import evaluation_demo


def evaluation_smoke(cfg: TaCConfig | None = None) -> dict[str, Any]:
    _ = cfg
    demo = evaluation_demo(seed=0, compression_ratio=4)
    return {
        "paper": "arXiv:2605.28713",
        "tac_vanilla_em": demo["tac_vanilla"]["em"],
        "tac_c_em": demo["tac_c"]["em"],
        "tac_c_hack_avoided": demo["tac_c"]["best_reward"]["hack_gate"] == 1.0,
        "best_utility": demo["tac_c"]["best_reward"]["utility"],
    }
