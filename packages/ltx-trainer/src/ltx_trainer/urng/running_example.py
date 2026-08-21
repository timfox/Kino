"""Fig. 2 running example (URNG pruning illustration)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.urng.intervals import Interval

# Intervals from Fig. 2; 2-D vectors chosen so triangle/RNG witnesses match paper cases.
FIG2_NODES: dict[str, dict[str, Any]] = {
    "A": {"interval": Interval(20, 80), "vector": (8.983719244171779, 4.1738737310306675)},
    "B": {"interval": Interval(30, 70), "vector": (1.0241941843441484, 4.850815596117094)},
    "C": {"interval": Interval(25, 75), "vector": (0.5796921703576992, -4.073076982579936)},
    "D": {"interval": Interval(30, 85), "vector": (4.734515173691747, -1.9751715201174473)},
    "E": {"interval": Interval(30, 100), "vector": (10.15756931607116, 2.106935571767)},
    "F": {"interval": Interval(0, 70), "vector": (-1.045558081719287, -4.682555776911609)},
    "G": {"interval": Interval(25, 75), "vector": (6.402198327214569, -4.7507234242824445)},
    "H": {"interval": Interval(40, 60), "vector": (6.99062108299389, 2.624745644808435)},
}

# Paper Example 1 edge semantics (Fig. 2 narrative)
EXAMPLE1_BITS: dict[tuple[str, str], tuple[int, int]] = {
    ("B", "C"): (1, 0),
    ("B", "D"): (0, 1),
    ("D", "G"): (1, 1),
}


def fig2_dataset() -> dict[str, Any]:
    return {
        "nodes": sorted(FIG2_NODES),
        "intervals": {k: [v["interval"].left, v["interval"].right] for k, v in FIG2_NODES.items()},
        "dim": len(next(iter(FIG2_NODES.values()))["vector"]),
    }
