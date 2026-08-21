"""Paper table anchors for PhysX-Omni."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"PhysX-Omni": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "PhysX-Bench: geometry, scale, material, affordance, kinematics, description",
        "website": "https://physx-omni.github.io/",
    }
