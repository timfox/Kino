"""PiD pipeline re-exports for paper stub smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pid.config import PiDConfig
from ltx_trainer.pid.evaluation import (
    evaluation_demo as _evaluation_demo,
    framework_card as _framework_card,
    knowledge_card as _knowledge_card,
)


def framework_card(cfg: PiDConfig | None = None) -> dict[str, Any]:
    return _framework_card()


def knowledge_card(cfg: PiDConfig | None = None) -> dict[str, Any]:
    return _knowledge_card()


def evaluation_demo(cfg: PiDConfig | None = None) -> dict[str, Any]:
    return _evaluation_demo()


def evaluation_smoke(cfg: PiDConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PiDConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.pid",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "shapes": ev.get("shapes"),
    }
