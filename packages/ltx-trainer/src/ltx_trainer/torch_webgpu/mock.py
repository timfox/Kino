"""Runnable evaluation smoke for torch-webgpu (arXiv:2604.02344)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.torch_webgpu.benchmarks import benchmarks_bundle
from ltx_trainer.torch_webgpu.pipeline import evaluation_smoke as _pipeline_smoke


def evaluation_smoke() -> dict[str, Any]:
    out = _pipeline_smoke(seed=0)
    out["benchmarks"] = benchmarks_bundle()
    return out
