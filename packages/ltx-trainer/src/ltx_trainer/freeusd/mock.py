"""Runnable evaluation smoke for FreeUSD."""

from __future__ import annotations

from typing import Any

from ltx_trainer.freeusd.pipeline import evaluation_smoke as _pipeline_smoke


def evaluation_smoke() -> dict[str, Any]:
    return _pipeline_smoke(seed=0)
