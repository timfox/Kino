"""AgenticVBench evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.agenticvbench.config import AgenticVBenchConfig
from ltx_trainer.agenticvbench.pipeline import evaluation_smoke as _evaluation_smoke


def evaluation_smoke(cfg: AgenticVBenchConfig | None = None) -> dict[str, Any]:
    return _evaluation_smoke(cfg)
