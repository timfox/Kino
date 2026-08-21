"""Framework card and benchmarks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.tree_eps_accept.config import TreeEpsAcceptConfig
from ltx_trainer.tree_eps_accept.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.tree_eps_accept.mock import evaluation_smoke
from ltx_trainer.tree_eps_accept.tables import (
    acceptance_game_table,
    bisimulation_distance_game_table,
    epsilon_acceptance_game_table,
    headline_results,
)


def framework_card(cfg: TreeEpsAcceptConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TreeEpsAcceptConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": cfg.authors,
        "problem": (
            "Rigid tree automata acceptance is Boolean; ε-acceptance games admit a "
            "measurable defect budget via distance liftings on binary trees."
        ),
        "main_result": cfg.main_theorem,
        "discipline_stack": [
            "Prompt / context engineering (companion augment-engineering line)",
            "Rigid acceptance games (Kupke–Venema coalgebraic automata)",
            "ε-acceptance + bisimulation distance (this paper)",
        ],
        "games": {
            "acceptance": [r["position"] for r in acceptance_game_table()],
            "bisimulation_distance": [r["position"] for r in bisimulation_distance_game_table()],
            "epsilon_acceptance": [r["position"] for r in epsilon_acceptance_game_table()],
        },
        "section5_examples": list(cfg.section5_examples),
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "acceptance_game": acceptance_game_table(),
        "bisimulation_distance_game": bisimulation_distance_game_table(),
        "epsilon_acceptance_game": epsilon_acceptance_game_table(),
        "headlines": headline_results(),
    }


def evaluation_demo() -> dict[str, Any]:
    return evaluation_smoke()
