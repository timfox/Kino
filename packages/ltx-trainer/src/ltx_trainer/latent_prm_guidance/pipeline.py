"""End-to-end demo: latent branches + PRM selection + reward."""

from __future__ import annotations

from typing import Any

from ltx_trainer.latent_prm_guidance.benchmarks import summary_anchors, table_1_main
from ltx_trainer.latent_prm_guidance.config import LatentPrmGuidanceConfig
from ltx_trainer.latent_prm_guidance.constants import TABLE1_VALIDATION
from ltx_trainer.latent_prm_guidance.latent import greedy_branch_select, sample_branches
from ltx_trainer.latent_prm_guidance.prm import prefix_target_mean
from ltx_trainer.latent_prm_guidance.reward import terminal_reward


def _stub_prm_score(prefix_len: int, branch_idx: int, *, unperturbed_idx: int = 0) -> float:
    """Deterministic stub: prefer unperturbed branch, penalize large perturbation index."""
    if branch_idx == unperturbed_idx:
        return 0.92
    return max(0.15, 0.85 - 0.08 * branch_idx)


def run_demo(cfg: LatentPrmGuidanceConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentPrmGuidanceConfig()
    h_t = [0.1] * 8  # toy hidden state
    branches = sample_branches(h_t, cfg.branches_test, seed=42)
    scores = [(i, _stub_prm_score(3, i)) for i in range(len(branches))]
    selected = greedy_branch_select(scores)

    # Toy rollout targets for one latent step.
    rollout_rewards = [
        terminal_reward(compiled=True, executed=True, validators={"validation": 1.0}),
        terminal_reward(compiled=True, executed=False),
        terminal_reward(compiled=False, executed=False),
    ]
    rt = prefix_target_mean(rollout_rewards)

    latent_row = next(r for r in TABLE1_VALIDATION if r["method"] == "Latent reasoning")
    prm_row = next(r for r in TABLE1_VALIDATION if r["method"] == "Latent PRM guidance")
    gain = float(prm_row["no_repair"]) - float(latent_row["no_repair"])

    return {
        "config": {
            "primary_model": cfg.primary_model,
            "prm_backbone": cfg.prm_backbone,
            "latent_steps": cfg.latent_steps,
            "branches_test": cfg.branches_test,
        },
        "branch_demo": {
            "n_candidates": len(branches),
            "selected_branch": selected,
            "top_score": scores[selected][1],
        },
        "prefix_target_rt": rt,
        "table_1": table_1_main(),
        "prm_beats_latent_unguided": gain > 8.0,
        "summary": summary_anchors(),
    }
