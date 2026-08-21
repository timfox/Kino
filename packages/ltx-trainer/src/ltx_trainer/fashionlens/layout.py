"""Scope notes for FashionLens reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No MLLM training or U-FIRE data loading: PGSQC, GGAS, and InfoNCE are toy math only.",
    "Qwen3-VL backbone, LoRA, GradCache, and VERL-style distributed training are not run here.",
    "Table II mR values are quoted from the paper for agent-facing benchmarks, not reproduced.",
)
