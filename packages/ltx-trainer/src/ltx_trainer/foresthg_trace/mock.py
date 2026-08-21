"""ForestHG-Trace evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.foresthg_trace.config import ForestHGConfig
from ltx_trainer.foresthg_trace.pipeline import evaluation_smoke as _evaluation_smoke


def evaluation_smoke(cfg: ForestHGConfig | None = None) -> dict[str, Any]:
    return _evaluation_smoke(cfg)
