"""Paper table stubs for MCB."""

from __future__ import annotations

from ltx_trainer.mcb.baselines import (
    DEEPSEEK_INDEPENDENCE,
    DEVELOPER_SPLIT,
    FABRICATION,
    FRAMING_LIFT,
    PAPER_ANCHORS,
)
from ltx_trainer.mcb.pipeline import evaluation_smoke


def table_developer_split() -> dict:
    return dict(DEVELOPER_SPLIT)


def table_fabrication() -> dict:
    return dict(FABRICATION)


def table_framing() -> dict:
    return dict(FRAMING_LIFT)


def table_deepseek() -> dict:
    return dict(DEEPSEEK_INDEPENDENCE)


def anchors() -> dict:
    return dict(PAPER_ANCHORS)


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke()
