"""Framework card and DuplexSLA-Bench table excerpts (arXiv:2605.20755)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.duplex_sla.config import DuplexSlaConfig
from ltx_trainer.duplex_sla.layout import LIMITATIONS
from ltx_trainer.duplex_sla.mock import evaluation_smoke


def table1_design_summary() -> list[dict[str, Any]]:
    """Table 1 — system-level summary."""
    return [
        {"element": "Backbone scale", "formulation": "7B speech-LM, initialized from Step-Audio 2 mini"},
        {"element": "Streaming clock", "formulation": "160 ms conversational chunks"},
        {"element": "User audio granularity", "formulation": "2 causal acoustic features per chunk (80 ms each)"},
        {"element": "Assistant audio granularity", "formulation": "4 discrete audio tokens per chunk (40 ms each)"},
        {"element": "Per-chunk speech layout", "formulation": "TA4 (one text anchor + four audio tokens)"},
        {
            "element": "Action channel content",
            "formulation": "Delayed transcript, planning text, turn-taking labels, tool calls",
        },
        {"element": "Per-chunk action token budget", "formulation": "≤ 10 tokens; overflow spills into next chunks"},
        {"element": "Tool-call schema", "formulation": "50 cabin and smart-home function schemas + 3 interaction-control labels"},
        {"element": "Native duplex behaviours", "formulation": "Pause, interrupt, backchannel without external semantic VAD"},
        {"element": "Online tool calling", "formulation": "Backchannel-triggered, single-action, multi-action"},
    ]


def table2_training_mixture() -> list[dict[str, Any]]:
    """Table 2 — CPT vs post-training audio hours (approximate scales from paper)."""
    return [
        {"stage": "CPT — Text", "scale": "~1.92 M samples"},
        {"stage": "CPT — Duplex dialogue", "scale": "~320 k hours"},
        {"stage": "CPT — User-channel ASR", "scale": "~90 k hours"},
        {"stage": "CPT — Assistant-channel ASR", "scale": "~90 k hours"},
        {"stage": "Post — Interrupt + backchannel + pause", "scale": "~36 k hours"},
        {"stage": "Post — Tool-call (BC-action, single, multi)", "scale": "~14 k hours"},
    ]


def table3_turn_taking_eval_protocol() -> list[dict[str, Any]]:
    """Table 3 — accuracy windows and delay anchors for DuplexSLA-Bench turn-taking."""
    return [
        {
            "scenario": "normal",
            "accuracy_window": "assistant speech onset ∈ [t_ue − 0.2, +∞)",
            "delay_definition": "|t_speak − t_ue|",
        },
        {
            "scenario": "pause",
            "accuracy_window": "same as normal on hesitation-rich user audio",
            "delay_definition": "|t_speak − t_ue|",
        },
        {
            "scenario": "interrupt",
            "accuracy_window": "assistant stop time ∈ [t_int − 1, t_int + 2]",
            "delay_definition": "|t_stop − t_int|",
        },
        {
            "scenario": "backchannel",
            "accuracy_window": "stop-or-restart event ∈ [t_bc_s − 0.2, t_bc_e + 2]",
            "delay_definition": "|t_label − t_bc_e|",
        },
    ]


def table_duplexsla_bench_composition() -> list[dict[str, Any]]:
    """Table 10 style — DuplexSLA-Bench subset counts."""
    return [
        {"subset": "Turn-taking — normal", "cases": 300},
        {"subset": "Turn-taking — pause", "cases": 300},
        {"subset": "Turn-taking — interrupt", "cases": 300},
        {"subset": "Turn-taking — backchannel", "cases": 300},
        {"subset": "Tool-call — single-action", "cases": 300},
        {"subset": "Tool-call — multi-action", "cases": 300},
        {"subset": "Tool-call — backchannel-action", "cases": 300},
    ]


def table5_tool_call_results() -> list[dict[str, Any]]:
    """Table 5 — tool-call subset (900 cases), average and per-pattern."""
    return [
        {
            "model": "ASR + LLM sys",
            "avg_acc_pct": 91.33,
            "avg_delay_s": 2.77,
            "single_acc_pct": 89.33,
            "single_delay_s": 2.33,
            "multi_acc_pct": 89.33,
            "multi_delay_s": 4.71,
            "bc_acc_pct": 95.33,
            "bc_delay_s": 1.27,
        },
        {
            "model": "DuplexSLA",
            "avg_acc_pct": 85.56,
            "avg_delay_s": 0.64,
            "single_acc_pct": 85.67,
            "single_delay_s": 0.67,
            "multi_acc_pct": 75.00,
            "multi_delay_s": 0.68,
            "bc_acc_pct": 96.00,
            "bc_delay_s": 0.57,
        },
    ]


def table6_context_prefill_turn_taking() -> list[dict[str, Any]]:
    """Table 6 — context-prefill setting (four scenarios)."""
    return [
        {
            "model": "DuplexSLA",
            "normal_acc_pct": 96.00,
            "normal_delay_s": 0.27,
            "pause_acc_pct": 93.33,
            "pause_delay_s": 0.27,
            "interrupt_acc_pct": 99.33,
            "interrupt_delay_s": 0.40,
            "backchannel_acc_pct": 98.33,
            "backchannel_delay_s": 0.32,
        },
        {
            "model": "gemini-3.1-flash-live",
            "normal_acc_pct": 93.67,
            "normal_delay_s": 1.18,
            "pause_acc_pct": 94.33,
            "pause_delay_s": 1.17,
            "interrupt_acc_pct": 63.67,
            "interrupt_delay_s": 0.62,
            "backchannel_acc_pct": 40.00,
            "backchannel_delay_s": None,
        },
        {
            "model": "gpt-realtime-1.5 (semantic-vad-high)",
            "normal_acc_pct": 91.33,
            "normal_delay_s": 1.67,
            "pause_acc_pct": 90.33,
            "pause_delay_s": 1.68,
            "interrupt_acc_pct": 79.00,
            "interrupt_delay_s": 0.68,
            "backchannel_acc_pct": 0.33,
            "backchannel_delay_s": None,
        },
        {
            "model": "gpt-realtime-1.5 (server-vad-40ms)",
            "normal_acc_pct": 82.33,
            "normal_delay_s": 0.95,
            "pause_acc_pct": 71.00,
            "pause_delay_s": 1.02,
            "interrupt_acc_pct": 77.00,
            "interrupt_delay_s": 0.72,
            "backchannel_acc_pct": 13.00,
            "backchannel_delay_s": None,
        },
    ]


def table7_no_context_prefill() -> list[dict[str, Any]]:
    """Table 7 — no-prefill (normal + pause only)."""
    return [
        {"model": "DuplexSLA", "avg_acc_pct": 94.34, "avg_delay_s": 0.30, "normal_acc_pct": 95.67, "normal_delay_s": 0.29, "pause_acc_pct": 93.00, "pause_delay_s": 0.31},
        {"model": "Freeze-Omni", "avg_acc_pct": 10.67, "avg_delay_s": 0.36, "normal_acc_pct": 10.33, "normal_delay_s": 0.40, "pause_acc_pct": 11.00, "pause_delay_s": 0.33},
        {"model": "PersonaPlex", "avg_acc_pct": 22.34, "avg_delay_s": 0.47, "normal_acc_pct": 22.67, "normal_delay_s": 0.38, "pause_acc_pct": 22.00, "pause_delay_s": 0.55},
        {"model": "MiniCPM-o", "avg_acc_pct": 82.00, "avg_delay_s": 0.61, "normal_acc_pct": 83.33, "normal_delay_s": 0.62, "pause_acc_pct": 80.67, "pause_delay_s": 0.59},
        {"model": "gemini-3.1-flash-live", "avg_acc_pct": 93.17, "avg_delay_s": 1.17, "normal_acc_pct": 93.67, "normal_delay_s": 1.16, "pause_acc_pct": 93.67, "pause_delay_s": 1.18},
        {
            "model": "gpt-realtime-1.5 (semantic-vad-high)",
            "avg_acc_pct": 96.50,
            "avg_delay_s": 1.57,
            "normal_acc_pct": 96.70,
            "normal_delay_s": 1.57,
            "pause_acc_pct": 96.30,
            "pause_delay_s": 1.57,
        },
        {
            "model": "gpt-realtime-1.5 (server-vad-40ms)",
            "avg_acc_pct": 85.50,
            "avg_delay_s": 0.83,
            "normal_acc_pct": 91.30,
            "normal_delay_s": 0.83,
            "pause_acc_pct": 79.70,
            "pause_delay_s": 0.83,
        },
    ]


def framework_card(cfg: DuplexSlaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DuplexSlaConfig()
    return {
        "name": cfg.title,
        "paper": cfg.paper_arxiv,
        "authors": "Haoyang Zhang, Jun Chen, Donghang Wu, Yuxin Li, Yuxin Zhang, Xiangyu Tony Zhang, Che Liu, Qingjian Lin, Yizhou Peng, Hexin Liu, Eng Siong Chng, Chao Yan, Boyong Wu, Yechang Huang, Xuerui Yang, Fei Tian (StepFun, PKU, NTU, and affiliates)",
        "repo": cfg.repo,
        "problem": "Turn-based cascades and two-stream duplex lack a native time-aligned lane for planning and tool calls.",
        "formulation": {
            "channels": "Continuous user audio; discrete assistant TA4 (1 text anchor + 4 audio codes per chunk); rate-limited action text.",
            "clock": f"{cfg.chunk_ms} ms chunks; user {cfg.user_features_per_chunk}×{cfg.user_feature_stride_ms} ms; assistant {cfg.assistant_audio_tokens_per_chunk}×{cfg.assistant_audio_stride_ms} ms.",
            "serialization": "<|user_audio_begin|> U U <|user_audio_end|> <|assistant_audio_begin|> T A A A A <|assistant_audio_end|> ⟨action⟩ <|action_end|>",
            "action_cap": f"≤ {cfg.action_tokens_max_per_chunk} action tokens per chunk; FIFO spill across chunks; atomic <|toolcall_begin|>…<|toolcall_end|> blocks.",
        },
        "capabilities": [
            "Semantic-driven pause, interrupt, and backchannel on the action channel (no external semantic VAD).",
            "In-conversation planning plus JSON tool calls while assistant TA4 keeps speaking.",
        ],
        "benchmark": f"DuplexSLA-Bench: {cfg.bench_total_cases} cases ({cfg.bench_turn_taking_cases} turn-taking + {cfg.bench_tool_cases} tool-call).",
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }


def headline_results(cfg: DuplexSlaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DuplexSlaConfig()
    return {
        "bench_total_cases": cfg.bench_total_cases,
        "tool_avg_delay_duplex_s": cfg.tool_duplex_avg_delay_s,
        "tool_avg_delay_cascade_s": cfg.tool_cascade_avg_delay_s,
        "duplex_backchannel_acc_pct": 98.33,
        "no_prefill_avg_delay_s": 0.30,
    }


def evaluation_demo(cfg: DuplexSlaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DuplexSlaConfig()
    return {"paper": cfg.paper_arxiv, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle(cfg: DuplexSlaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DuplexSlaConfig()
    return {
        "framework": framework_card(cfg),
        "table1_design": table1_design_summary(),
        "table2_training_mix": table2_training_mixture(),
        "table3_turn_taking_protocol": table3_turn_taking_eval_protocol(),
        "duplexsla_bench_composition": table_duplexsla_bench_composition(),
        "table5_tool_calls": table5_tool_call_results(),
        "table6_prefill": table6_context_prefill_turn_taking(),
        "table7_no_prefill": table7_no_context_prefill(),
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }
