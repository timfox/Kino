"""UB-spec §7.3 opt-in ordering surface (arXiv:2605.28717)."""

from __future__ import annotations

from typing import Any

# Service modes (wire spreading) × execution tags (issue gating)
SERVICE_MODES = ("ROI", "ROT", "ROL", "UNO")
EXECUTION_TAGS = ("NO", "RO", "SO")
COMPLETION_ORDER = ("arrival", "issue")


def ordering_surface_card() -> dict[str, Any]:
    return {
        "service_modes": list(SERVICE_MODES),
        "execution_tags": list(EXECUTION_TAGS),
        "completion_order": list(COMPLETION_ORDER),
        "fence": "explicit application barrier",
        "cold_path_cycles": 24,
        "gating_max_cycles": 50,
        "note": "All 12 mode×tag combos emit first wire flit at 24 cy; gating paid only when requested.",
    }


def gating_cost_stub(pending_deps: int, *, fenced: bool = False, strict_order: bool = False) -> dict[str, Any]:
    """Synthetic gating delay after completion notification."""
    if not fenced and not strict_order:
        return {"cycles": 0, "gated": False}
    cycles = min(48, 4 + pending_deps * 8 + (12 if strict_order else 0))
    return {"cycles": cycles, "gated": True}
