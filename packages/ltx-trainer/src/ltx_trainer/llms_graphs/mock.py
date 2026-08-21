"""Runnable evaluation smoke for LLMs+Graphs (arXiv:2606.11560)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.llms_graphs.benchmarks import benchmarks_bundle
from ltx_trainer.llms_graphs.pipeline import evaluation_smoke as _pipeline_smoke


def evaluation_smoke() -> dict[str, Any]:
    out = _pipeline_smoke(seed=0)
    out["benchmarks"] = benchmarks_bundle()
    return out
