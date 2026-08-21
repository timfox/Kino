"""Environment gates for Dyna-Pruner integration."""

from __future__ import annotations

import os

from dataclasses import replace

from ltx_trainer.dyna_pruner.config import DynaPrunerConfig


def dyna_pruner_enabled() -> bool:
    return os.environ.get("GOPEX_DYNA_PRUNER", "").strip().lower() in ("1", "true", "yes")


def read_config() -> DynaPrunerConfig:
    cfg = DynaPrunerConfig()
    sd = os.environ.get("GOPEX_DYNA_PRUNER_SD")
    sw = os.environ.get("GOPEX_DYNA_PRUNER_SW")
    if sd is not None and sd.strip():
        cfg = replace(cfg, data_sparsity_sd=float(sd))
    if sw is not None and sw.strip():
        cfg = replace(cfg, model_sparsity_sw=float(sw))
    return cfg


def num_transformer_blocks(default: int = 48) -> int:
    raw = os.environ.get("GOPEX_DYNA_PRUNER_NUM_BLOCKS", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return default
