"""Structured DS workflow state and cost-aware routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DSState:
    """Definition 1: St = {Tt, Dt, Pt, Lt}."""

    task: str
    data_stats: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    progress: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "data_stats": dict(self.data_stats),
            "env": dict(self.env),
            "logs": list(self.logs),
            "progress": self.progress,
        }


def construct_state(
    *,
    task: str,
    n_rows: int = 1000,
    n_cols: int = 10,
    libraries: list[str] | None = None,
    last_error: str | None = None,
) -> DSState:
    """State Constructor SC(E) → structured St."""
    logs = []
    if last_error:
        logs.append(f"error: {last_error}")
    return DSState(
        task=task,
        data_stats={"n_rows": n_rows, "n_cols": n_cols, "preview_rows": min(5, n_rows)},
        env={"libraries": libraries or ["numpy", "pandas", "sklearn"]},
        logs=logs,
        progress="ready",
    )


@dataclass
class RouteDecision:
    mode: str  # execute | simulate
    estimated_cost_min: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "estimated_cost_min": round(self.estimated_cost_min, 3),
            "reason": self.reason,
        }


# Word-boundary-ish heavy ops (avoid matching filenames like train.csv).
_HEAVY_PATTERNS = (
    "fit(",
    ".fit(",
    "model.fit",
    "cross_val",
    "gridsearch",
    "optuna",
    "xgboost",
    "lightgbm",
    "lgbm",
    "epoch",
    ".train(",
    "trainer.",
)


def estimate_action_cost(action: str) -> float:
    """Proxy minutes for cost-aware routing."""
    a = action.lower()
    cost = 0.1
    for kw in _HEAVY_PATTERNS:
        if kw in a:
            cost += 8.0
            break
    if "read_csv" in a or "head(" in a or "describe(" in a:
        cost += 0.2
    return cost


def route_action(
    state: DSState,
    action: str,
    *,
    cost_threshold: float = 5.0,
    force_timeout: bool = False,
) -> RouteDecision:
    """Router R(St, At) → execute | simulate (Eq. 4)."""
    if force_timeout:
        return RouteDecision("simulate", 999.0, "compiler_timeout_redirect")
    cost = estimate_action_cost(action)
    if cost >= cost_threshold:
        return RouteDecision("simulate", cost, "expensive_action")
    return RouteDecision("execute", cost, "lightweight_action")
