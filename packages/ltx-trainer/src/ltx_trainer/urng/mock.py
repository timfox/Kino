"""Runnable evaluation smoke for URNG / UG (arXiv:2606.11789)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.urng.benchmarks import benchmarks_bundle
from ltx_trainer.urng.pipeline import evaluation_smoke as _pipeline_smoke


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    out = _pipeline_smoke(seed=seed)
    out["benchmarks"] = benchmarks_bundle()
    return out
