"""VCap evaluation smoke for paper stub runner."""

from __future__ import annotations

from typing import Any

from ltx_trainer.vcap.config import VCapConfig
from ltx_trainer.vcap.pipeline import evaluation_smoke as _evaluation_smoke


def evaluation_smoke(cfg: VCapConfig | None = None) -> dict[str, Any]:
    return _evaluation_smoke(cfg)
