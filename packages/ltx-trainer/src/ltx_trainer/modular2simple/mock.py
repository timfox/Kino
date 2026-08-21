"""Runnable evaluation smoke for Modular2Simple."""

from __future__ import annotations

from typing import Any

from ltx_trainer.modular2simple.benchmarks import benchmarks_bundle
from ltx_trainer.modular2simple.pipeline import evaluation_smoke as _pipeline_smoke


def evaluation_smoke() -> dict[str, Any]:
    out = _pipeline_smoke(seed=0)
    out["benchmarks"] = benchmarks_bundle()
    return out
