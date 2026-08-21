"""Eval helpers."""

from __future__ import annotations

from ltx_trainer.aac_audioset.mock import evaluation_smoke
from ltx_trainer.aac_audioset.pipeline import pipeline_demo


def eval_smoke() -> dict:
    return evaluation_smoke()


def pipeline_demo_export(seed: int = 42) -> dict:
    return pipeline_demo(seed=seed)
