"""Group-action SE(2) composition smoke (arXiv:2605.24578)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.group_action.config import GroupActionConfig


def _accumulate_actions(actions: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    sx = sy = st = 0.0
    for dx, dy, dt in actions:
        sx += dx
        sy += dy
        st += dt
    return (sx, sy, st)


def evaluation_smoke(cfg: GroupActionConfig | None = None) -> dict[str, Any]:
    c = cfg or GroupActionConfig()
    forward = [(0.1, 0.0, 0.05), (0.2, 0.0, 0.05)]
    inv = [(-dx, -dy, -dt) for dx, dy, dt in reversed(forward)]
    acc = _accumulate_actions(forward)
    return {
        "paper": "arXiv:2605.24578",
        "accumulated_dx": round(acc[0], 4),
        "inverse_steps": len(inv),
        "lambda_gar": getattr(c, "lambda_gar", 1.0),
    }
