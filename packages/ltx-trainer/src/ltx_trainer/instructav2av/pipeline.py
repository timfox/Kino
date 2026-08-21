"""Pipeline re-exports for paper stub smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.instructav2av.config import InstructAV2AVConfig
from ltx_trainer.instructav2av.evaluation import (
    evaluation_demo as _evaluation_demo,
    framework_card as _framework_card,
    knowledge_card as _knowledge_card,
)


def framework_card(cfg: InstructAV2AVConfig | None = None) -> dict[str, Any]:
    return _framework_card()


def knowledge_card(cfg: InstructAV2AVConfig | None = None) -> dict[str, Any]:
    return _knowledge_card()


def evaluation_demo(cfg: InstructAV2AVConfig | None = None) -> dict[str, Any]:
    return _evaluation_demo()


def evaluation_smoke(cfg: InstructAV2AVConfig | None = None) -> dict[str, Any]:
    cfg = cfg or InstructAV2AVConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.instructav2av",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "siga_gate_mean": ev["siga"]["gate_mean"],
    }
