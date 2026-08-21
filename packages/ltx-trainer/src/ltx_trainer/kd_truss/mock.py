"""Runnable evaluation smoke for (k, δ)-truss (arXiv:2606.11582)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.kd_truss.benchmarks import benchmarks_bundle
from ltx_trainer.kd_truss.pipeline import evaluation_smoke as _pipeline_smoke


def evaluation_smoke() -> dict[str, Any]:
    out = _pipeline_smoke(seed=0)
    out["benchmarks"] = benchmarks_bundle()
    return out
