"""Toy DQN-style partitioning agent: composite loss and top-N split pruning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.vtm_partition.config import VTMPartitionConfig
from ltx_trainer.vtm_partition.qtmtt import ALL_SPLITS


@dataclass
class PartitionActionCosts:
    """Ground-truth or predicted RD costs per split mode at one tree level."""

    costs: dict[str, float]
    syntax_cost_split: float = 0.0


def mse_level1(
    predicted: dict[str, float],
    ground_truth: dict[str, float],
) -> float:
    """Eq. (4) — MSE1 over root CU split actions."""
    errs = [(predicted[k] - ground_truth[k]) ** 2 for k in ALL_SPLITS if k in ground_truth]
    return float(np.mean(errs)) if errs else 0.0


def mse_level2(
    predicted_sub: dict[str, float],
    ground_truth_sub: dict[str, float],
) -> float:
    """Eq. (5) — MSE2 over sub-CU costs given root action."""
    return mse_level1(predicted_sub, ground_truth_sub)


def mse_hierarchical_td(
    predicted_parent: float,
    predicted_children: dict[str, float],
    syntax_cost_split: float,
) -> float:
    """Eq. (6) — Bellman-like consistency between parent Q and children."""
    child_sum = sum(predicted_children.values())
    residual = predicted_parent - (child_sum + syntax_cost_split)
    return float(residual**2)


def composite_loss(
    *,
    pred_l1: dict[str, float],
    gt_l1: dict[str, float],
    pred_l2: dict[str, float],
    gt_l2: dict[str, float],
    pred_parent: float,
    syntax_cost: float,
    alphas: tuple[float, float, float] = (1.0, 1.0, 0.5),
) -> float:
    """Eq. (3): L = α1·MSE1 + α2·MSE2 + α3·MSE3."""
    a1, a2, a3 = alphas
    l1 = mse_level1(pred_l1, gt_l1)
    l2 = mse_level2(pred_l2, gt_l2)
    l3 = mse_hierarchical_td(pred_parent, pred_l2, syntax_cost)
    return a1 * l1 + a2 * l2 + a3 * l3


def predict_q_values_linear(state: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Deterministic toy Q-head: affine map from state to per-split costs."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((len(ALL_SPLITS), state.size))
    b = rng.standard_normal(len(ALL_SPLITS)) * 0.1
    q = w @ state + b
    return {mode: float(q[i]) for i, mode in enumerate(ALL_SPLITS)}


def select_splits_top_n(
    q_values: dict[str, float],
    *,
    top_n: int = 3,
    threshold: float | None = None,
    always_include_ns: bool = True,
) -> list[str]:
    """
  Inference: evaluate only top-N splits with lowest predicted Q (RD cost).

  Optional threshold T restricts to modes within T of the minimum Q.
  """
    ranked = sorted(q_values.items(), key=lambda kv: kv[1])
    min_q = ranked[0][1]
    chosen: list[str] = []
    for mode, q in ranked:
        if len(chosen) >= top_n:
            break
        if threshold is not None and q > min_q * (1.0 + threshold) and mode != "NS":
            continue
        chosen.append(mode)
    if always_include_ns and "NS" not in chosen:
        chosen = ["NS"] + chosen[:-1] if len(chosen) >= top_n else ["NS"] + chosen
    return chosen


def rl_tradeoff_point(
    *,
    top_n: int,
    max_mtt_depth: int = 3,
    cfg: VTMPartitionConfig | None = None,
) -> dict[str, float | int | str]:
    """Lookup a paper Table 3 row by matching top-N and MTT depth config."""
    cfg = cfg or VTMPartitionConfig()
    key = f"{top_n}{max_mtt_depth}{max_mtt_depth}"
    rows = {r["nrl_config"]: r for r in table_iii_rl_tradeoffs()}
    if key in rows:
        return rows[key]
    # Interpolate stub: fewer splits evaluated → more speed, higher BD-rate
    base = rows.get("333", rows[list(rows.keys())[0]])
    speed_bonus = (6 - top_n) * 8.0
    return {
        "nrl_config": key,
        "bd_rate_pct": float(base["bd_rate_pct"]) + max(0, 2 - top_n) * 0.4,
        "et_pct": min(110.0, float(base["et_pct"]) + speed_bonus),
        "pixel_ratio_pct": min(98.0, float(base["pixel_ratio_pct"]) + speed_bonus * 0.6),
        "cu_ratio_pct": min(98.0, float(base["cu_ratio_pct"]) + speed_bonus * 0.5),
    }


def table_iii_rl_tradeoffs() -> list[dict[str, float | int | str]]:
    """Table 3 — RL agent trade-offs vs VTM-18.0 AI (paper)."""
    return [
        {
            "nrl_config": "111",
            "bd_rate_pct": 2.42,
            "et_pct": 27.60,
            "pixel_ratio_pct": 25.16,
            "cu_ratio_pct": 15.59,
        },
        {
            "nrl_config": "333",
            "bd_rate_pct": 1.81,
            "et_pct": 43.70,
            "pixel_ratio_pct": 28.56,
            "cu_ratio_pct": 34.89,
        },
        {
            "nrl_config": "222",
            "bd_rate_pct": 1.14,
            "et_pct": 48.60,
            "pixel_ratio_pct": 43.01,
            "cu_ratio_pct": 36.93,
        },
        {
            "nrl_config": "333_fast",
            "bd_rate_pct": 0.96,
            "et_pct": 54.40,
            "pixel_ratio_pct": 49.65,
            "cu_ratio_pct": 42.07,
        },
        {
            "nrl_config": "333_mid",
            "bd_rate_pct": 0.71,
            "et_pct": 67.60,
            "pixel_ratio_pct": 54.07,
            "cu_ratio_pct": 58.96,
        },
        {
            "nrl_config": "333_slow",
            "bd_rate_pct": 0.21,
            "et_pct": 89.90,
            "pixel_ratio_pct": 79.06,
            "cu_ratio_pct": 80.47,
        },
        {
            "nrl_config": "333_ref",
            "bd_rate_pct": 0.01,
            "et_pct": 105.50,
            "pixel_ratio_pct": 96.26,
            "cu_ratio_pct": 96.27,
        },
    ]
