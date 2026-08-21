"""OmniCustom: sync audio-video customization (arXiv:2602.12304)."""

from ltx_trainer.omnicustom.benchmark import BENCHMARK_CASES, BenchmarkCase, case_by_id
from ltx_trainer.omnicustom.config import OmniCustomConfig
from ltx_trainer.omnicustom.loss import OmniCustomLossBreakdown, total_omnicustom_loss
from ltx_trainer.omnicustom.mock import evaluation_smoke
from ltx_trainer.omnicustom.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
)
from ltx_trainer.omnicustom.prompts import ParsedPrompt, extract_speech, strip_speech_tags
from ltx_trainer.omnicustom.tables import appendix_lse_comparison, table2_quantitative, table3_user_study
from ltx_trainer.omnicustom.taxonomy import CustomizationSetting, SETTING_TRAITS

__all__ = [
    "BENCHMARK_CASES",
    "BenchmarkCase",
    "CustomizationSetting",
    "OmniCustomConfig",
    "OmniCustomLossBreakdown",
    "ParsedPrompt",
    "SETTING_TRAITS",
    "appendix_lse_comparison",
    "benchmarks_bundle",
    "case_by_id",
    "evaluation_demo",
    "evaluation_smoke",
    "extract_speech",
    "framework_card",
    "knowledge_card",
    "strip_speech_tags",
    "table2_quantitative",
    "table3_user_study",
    "total_omnicustom_loss",
]
