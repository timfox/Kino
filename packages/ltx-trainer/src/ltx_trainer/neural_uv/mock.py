"""Runnable evaluation smoke for Neural UV Atlas (arXiv:2606.10050)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.neural_uv.benchmarks import benchmarks_bundle
from ltx_trainer.neural_uv.pipeline import evaluation_smoke as _pipeline_smoke


def evaluation_smoke() -> dict[str, Any]:
    out = _pipeline_smoke(seed=0)
    out["benchmarks"] = benchmarks_bundle()
    return out
