"""Eval export helpers."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cs_asr_generalize.mock import evaluation_smoke
from ltx_trainer.cs_asr_generalize.pipeline import pipeline_demo


def eval_smoke() -> dict[str, Any]:
    return evaluation_smoke()


def pipeline_demo_export(seed: int = 42) -> dict[str, Any]:
    return pipeline_demo(seed=seed)
