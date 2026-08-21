"""Paper stub entry point for validate_paper_stubs / run_paper_stub_smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.eigenet.config import EigeNetConfig
from ltx_trainer.eigenet.pipeline import evaluation_smoke as _evaluation_smoke


def evaluation_smoke(cfg: EigeNetConfig | None = None) -> dict[str, Any]:
    return _evaluation_smoke(cfg)
