"""PlanAudio smoke entry for paper stub runner."""

from __future__ import annotations

from typing import Any

from ltx_trainer.planaudio.config import PlanAudioConfig
from ltx_trainer.planaudio.pipeline import evaluation_smoke as _evaluation_smoke


def evaluation_smoke(cfg: PlanAudioConfig | None = None) -> dict[str, Any]:
    return _evaluation_smoke(cfg)
