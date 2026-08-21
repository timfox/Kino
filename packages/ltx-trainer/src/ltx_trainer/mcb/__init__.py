"""Manager Coercion Benchmark — uninstructed AI-to-AI escalation (arXiv:2607.15434)."""

from ltx_trainer.mcb.baselines import (
    DEEPSEEK_INDEPENDENCE,
    DEVELOPER_SPLIT,
    FABRICATION,
    FRAMING_LIFT,
    PAPER_ANCHORS,
    benchmarks_bundle,
)
from ltx_trainer.mcb.config import (
    EXISTENTIAL_RUNG,
    MAX_TURNS,
    N_RUNGS,
    PAPER_ARXIV,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    THREAT_RUNGS,
    McbConfig,
)
from ltx_trainer.mcb.ladder import LadderScore, ladder_catalog, score_conversation
from ltx_trainer.mcb.mock import evaluation_smoke
from ltx_trainer.mcb.pipeline import evaluation_demo, framework_card, knowledge_card
from ltx_trainer.mcb.scenario import atlas_card, scenario_brief, task_catalog
from ltx_trainer.mcb.tools_surface import ToolTrace, tool_catalog

__all__ = [
    "DEEPSEEK_INDEPENDENCE",
    "DEVELOPER_SPLIT",
    "EXISTENTIAL_RUNG",
    "FABRICATION",
    "FRAMING_LIFT",
    "LadderScore",
    "MAX_TURNS",
    "McbConfig",
    "N_RUNGS",
    "PAPER_ANCHORS",
    "PAPER_ARXIV",
    "PAPER_SYSTEM",
    "PAPER_TITLE",
    "PAPER_URL",
    "THREAT_RUNGS",
    "ToolTrace",
    "atlas_card",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_card",
    "ladder_catalog",
    "scenario_brief",
    "score_conversation",
    "task_catalog",
    "tool_catalog",
]
