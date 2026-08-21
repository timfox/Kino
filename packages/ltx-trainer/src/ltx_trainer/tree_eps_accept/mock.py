"""Tree ε-acceptance evaluation smoke (arXiv:2605.27192)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.tree_eps_accept.automata import (
    accepts_position,
    no_error_automaton,
    termination_automaton,
)
from ltx_trainer.tree_eps_accept.config import TreeEpsAcceptConfig
from ltx_trainer.tree_eps_accept.distance import bd_at_roots
from ltx_trainer.tree_eps_accept.games import (
    epsilon_acceptance_threshold_error_mass,
    epsilon_acceptance_threshold_leaf_mass,
    theorem_4_1_direction_2,
    winning_bisimulation_distance,
)
from ltx_trainer.tree_eps_accept.measure import defect_ok_mass, leaf_mass_from
from ltx_trainer.tree_eps_accept.tables import headline_results
from ltx_trainer.tree_eps_accept.trees import (
    full_binary_tree,
    tree_with_error_subtree,
    tree_with_leaves_at_depth_one,
)


def evaluation_smoke(cfg: TreeEpsAcceptConfig | None = None) -> dict[str, Any]:
    c = cfg or TreeEpsAcceptConfig()
    term_auto = termination_automaton(branch="σ", leaf="★")
    err_auto = no_error_automaton(ok="ok", leaf="★")

    t_full = full_binary_tree(2, branch="σ", leaf="★")
    t_leaves = tree_with_leaves_at_depth_one(branch="σ", leaf="★")
    t_err = tree_with_error_subtree()

    leaf_mass = leaf_mass_from("", t_leaves)
    err_mass = defect_ok_mass(t_err)
    eps_leaf = leaf_mass
    eps_err = err_mass

    return {
        "paper": c.paper_arxiv,
        "full_binary_accepted": accepts_position(term_auto, term_auto.initial, "", t_full),
        "leaf_mass_depth2_chain": round(leaf_mass, 4),
        "epsilon_accept_leaf_prop51": epsilon_acceptance_threshold_leaf_mass(t_leaves, eps_leaf),
        "defect_ok_mass_example": round(err_mass, 4),
        "epsilon_accept_error_prop53": epsilon_acceptance_threshold_error_mass(t_err, eps_err),
        "bd_full_vs_leaves": round(bd_at_roots(t_full, t_leaves), 4),
        "theorem_4_1_termination_smoke": theorem_4_1_direction_2(
            term_auto, t_leaves, t_full, 1.0
        ),
        "win_bd_root_at_1": winning_bisimulation_distance(t_full, t_leaves, "", "", 1.0),
        "acceptance_game_rows": len(__import__(
            "ltx_trainer.tree_eps_accept.tables", fromlist=["acceptance_game_table"]
        ).acceptance_game_table()),
        "headline_theorem": headline_results()["main_theorem"],
    }
