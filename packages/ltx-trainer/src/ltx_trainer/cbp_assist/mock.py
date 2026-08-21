"""CBP assistance evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cbp_assist.config import CbpAssistConfig
from ltx_trainer.cbp_assist.pipeline import evaluation_smoke as _evaluation_smoke


def evaluation_smoke(cfg: CbpAssistConfig | None = None) -> dict[str, Any]:
    return _evaluation_smoke(cfg)
