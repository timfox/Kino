"""AUTOSCIENTISTS evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.autoscientists.config import AutoScientistsConfig
from ltx_trainer.autoscientists.pipeline import evaluation_smoke as _evaluation_smoke


def evaluation_smoke(cfg: AutoScientistsConfig | None = None) -> dict[str, Any]:
    return _evaluation_smoke(cfg)
