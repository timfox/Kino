"""DuplexSLA — full-duplex speech, language, and action on a shared chunk clock (arXiv:2605.20755)."""

from ltx_trainer.duplex_sla.action_queue import format_tool_call, spill_action_tokens
from ltx_trainer.duplex_sla.config import DuplexSlaConfig
from ltx_trainer.duplex_sla.layout import LIMITATIONS
from ltx_trainer.duplex_sla.mock import evaluation_smoke
from ltx_trainer.duplex_sla.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_design_summary,
    table2_training_mixture,
    table3_turn_taking_eval_protocol,
    table5_tool_call_results,
    table6_context_prefill_turn_taking,
    table7_no_context_prefill,
    table_duplexsla_bench_composition,
)

__all__ = [
    "LIMITATIONS",
    "DuplexSlaConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "format_tool_call",
    "framework_card",
    "headline_results",
    "spill_action_tokens",
    "table1_design_summary",
    "table2_training_mixture",
    "table3_turn_taking_eval_protocol",
    "table5_tool_call_results",
    "table6_context_prefill_turn_taking",
    "table7_no_context_prefill",
    "table_duplexsla_bench_composition",
]
