"""Virtual budget distribution — Algorithm 1 (Section IV-A)."""

from __future__ import annotations

from typing import Any


def _sorted_unique_latencies(latencies: list[float]) -> list[float]:
    return sorted(set(latencies), reverse=True)


def virtual_budget_distribution(
    latencies_by_accel: list[float],
    deadline: float,
) -> dict[str, Any]:
    """
    Algorithm 1: assign bm,ℓ with constraint levels ρ until Σ c↓(ρ) ≤ Dm.

    latencies_by_accel: distinct execution times on accelerators A for one layer.
    """
    levels = _sorted_unique_latencies(latencies_by_accel)
    r_max = len(levels)
    rho = 1
    while True:
        c_sel = levels[rho - 1]
        total = c_sel  # single-layer stub; multi-layer sums per model
        if total <= deadline:
            return {
                "status": "ok",
                "rho": rho,
                "selected_latency_us": c_sel,
                "budget_s": deadline * c_sel / total,
                "excluded_slower_than_us": levels[: rho - 1],
            }
        if rho >= r_max:
            return {"status": "infeasible", "rho": rho, "total_us": total, "deadline_s": deadline}
        rho += 1


def model_virtual_budgets(
    layer_latencies: list[list[float]],
    deadline: float,
) -> dict[str, Any]:
    """Assign per-layer budgets summing to model deadline (Eq. 1)."""
    budgets: list[float] = []
    rhos: list[int] = []
    for lat in layer_latencies:
        res = virtual_budget_distribution(lat, deadline / len(layer_latencies))
        if res["status"] != "ok":
            return {"status": "infeasible", "layer": len(budgets) + 1}
        budgets.append(res["budget_s"])
        rhos.append(res["rho"])
    total = sum(budgets)
    if abs(total - deadline) > 1e-9:
        scale = deadline / total
        budgets = [b * scale for b in budgets]
    return {
        "status": "ok",
        "budgets_s": [round(b, 6) for b in budgets],
        "constraint_levels": rhos,
        "deadline_s": deadline,
        "sum_budgets_s": round(sum(budgets), 6),
    }


def virtual_deadline(arrival_s: float, budgets_prefix: list[float]) -> list[float]:
    """Eq. 2: dv_j,m,ℓ = ta + Σ_{ℓ′≤ℓ} bm,ℓ′."""
    out = []
    acc = arrival_s
    for b in budgets_prefix:
        acc += b
        out.append(acc)
    return out
