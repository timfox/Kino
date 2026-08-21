"""MTAVG-Bench 2.0 evaluation smoke for paper stub runner."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mtavg2.config import MTAVG2Config
from ltx_trainer.mtavg2.pipeline import evaluation_smoke as _evaluation_smoke


def evaluation_smoke(cfg: MTAVG2Config | None = None) -> dict[str, Any]:
    return _evaluation_smoke(cfg)
