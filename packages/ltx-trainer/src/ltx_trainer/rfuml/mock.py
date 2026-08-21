"""Toy multi-view instances for R-FUML smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.rfuml.conflict import conflict_for_view
from ltx_trainer.rfuml.fuzzy import category_credibility, logits_to_memberships, uncertainty_from_credibility
from ltx_trainer.rfuml.fusion import fuse_memberships


@dataclass
class MultiViewInstance:
    logits_per_view: list[list[float]]
    true_label: list[float]


def make_clean_instance(*, num_views: int = 3, num_classes: int = 3) -> MultiViewInstance:
    base = [2.0, 0.2, 0.1]
    label = [1.0, 0.0, 0.0]
    return MultiViewInstance(logits_per_view=[base[:] for _ in range(num_views)], true_label=label)


def make_conflicting_instance(*, num_views: int = 3, num_classes: int = 3) -> MultiViewInstance:
    views = [
        [8.0, 0.1, 0.1],
        [8.0, 0.1, 0.1],
        [0.1, 0.1, 8.0],
    ]
    label = [1.0, 0.0, 0.0]
    return MultiViewInstance(logits_per_view=views[:num_views], true_label=label)


def instance_memberships(inst: MultiViewInstance) -> list[list[float]]:
    return [logits_to_memberships(v) for v in inst.logits_per_view]


def evaluation_smoke() -> dict[str, Any]:
    clean = make_clean_instance()
    conflict = make_conflicting_instance()
    mem_c = instance_memberships(clean)
    mem_x = instance_memberships(conflict)
    us_c = [uncertainty_from_credibility(category_credibility(m)) for m in mem_c]
    os_c = [conflict_for_view(mem_c, v) for v in range(len(mem_c))]
    us_x = [uncertainty_from_credibility(category_credibility(m)) for m in mem_x]
    os_x = [conflict_for_view(mem_x, v) for v in range(len(mem_x))]
    fused_c = fuse_memberships(mem_c, us_c, os_c, training=False)
    fused_x = fuse_memberships(mem_x, us_x, os_x, training=False)
    import math

    ent = sum(-p * math.log(p + 1e-12) for p in fused_x)
    return {
        "clean_fused_max": round(max(fused_c), 4),
        "conflict_fused_entropy": round(ent, 4),
    }
