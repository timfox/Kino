"""Lightweight eval helpers."""

from __future__ import annotations

from typing import Any

from ltx_trainer.qwen3_tts.config import Qwen3TtsConfig
from ltx_trainer.qwen3_tts.pipeline import evaluation_demo


def pipeline_demo(*, seed: int = 42, cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    return evaluation_demo(seed=seed, cfg=cfg)


def eval_smoke(cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    from ltx_trainer.qwen3_tts.mock import evaluation_smoke

    return evaluation_smoke(cfg)
