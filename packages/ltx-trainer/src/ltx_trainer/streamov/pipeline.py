"""StreamOV framework card, paper tables, and smoke demos (arXiv:2605.25621)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.streamov.config import StreamOVConfig
from ltx_trainer.streamov.evidence import build_multimodal_evidence
from ltx_trainer.streamov.memory import MemoryBank, update_long_short_memory
from ltx_trainer.streamov.trigger import ResponseTrigger


def framework_card(cfg: StreamOVConfig | None = None) -> dict[str, Any]:
    cfg = cfg or StreamOVConfig()
    return {
        "name": "StreamOV",
        "paper": cfg.paper_arxiv,
        "backbone": cfg.backbone,
        "frame_budget_sovbench": cfg.frame_budget_sovbench,
        "trigger_params_m": cfg.trigger_params_m,
        "mechanisms": [
            "Multimodal evidence routing (visual / audio / AV-aligned)",
            "Long-short term memory under fixed budget",
            "Hidden-state MLLM-as-trigger (no <silence> tokens, no external router)",
        ],
        "benchmarks": ["SOVBench-O", "SOVBench-T", "StreamingBench", "OVO-Bench"],
    }


def table_sovbench() -> dict[str, dict[str, float | None]]:
    """Table 1 — SOVBench (stream + QA context columns, avg + trigger F1)."""
    return {
        "Gemini 2.5 Flash": {
            "rt": 87.0, "recall": 80.4, "proactive": 66.8, "avg_stream": 75.6,
            "rt_qa": 88.9, "recall_qa": 85.6, "proactive_qa": 70.8, "avg_qa": 78.8, "f1": None,
        },
        "Qwen3-Omni-30B": {
            "rt": 85.3, "recall": 77.3, "proactive": 64.8, "avg_stream": 73.7,
            "rt_qa": 87.3, "recall_qa": 82.5, "proactive_qa": 74.3, "avg_qa": 79.9, "f1": None,
        },
        "ROMA-7B": {
            "rt": 78.0, "recall": 73.2, "proactive": 46.3, "avg_stream": 60.4,
            "rt_qa": 78.1, "recall_qa": 79.4, "proactive_qa": 47.5, "avg_qa": 61.5, "f1": 52.2,
        },
        "StreamOV": {
            "rt": 86.9, "recall": 73.2, "proactive": 78.6, "avg_stream": 81.6,
            "rt_qa": 86.7, "recall_qa": 80.4, "proactive_qa": 82.1, "avg_qa": 83.8, "f1": 90.5,
        },
    }


def table_streamingbench() -> dict[str, dict[str, float]]:
    """Table 2 — StreamingBench audio-visual and visual-only averages."""
    return {
        "Qwen3-Omni-30B": {"av_avg": 61.0, "vis_avg": 77.9},
        "ROMA-7B": {"av_avg": 46.1, "vis_avg": 72.4},
        "StreamOV": {"av_avg": 68.6, "vis_avg": 86.2},
    }


def table_ablation_components() -> dict[str, float]:
    """Table 4 — SOVBench-O ablation (avg accuracy %)."""
    return {
        "baseline": 73.7,
        "query_aware_only": 80.0,
        "query_agnostic_only": 78.0,
        "qa_plus_qag": 80.4,
        "full_with_long_memory": 81.6,
    }


def table_trigger_ablation() -> dict[str, dict[str, float]]:
    """Table 6 — trigger architecture F1 (%)."""
    return {
        "Qwen3omni + Trigger": {"f1": 81.4, "hidden_num": 1},
        "StreamOV + Trigger (h0)": {"f1": 90.5, "hidden_num": 1},
        "StreamOV + Trigger (h0,h1)": {"f1": 90.7, "hidden_num": 2},
    }


def training_step_demo(cfg: StreamOVConfig | None = None) -> dict[str, float]:
    """Smoke: evidence routing, memory update, trigger CE."""
    cfg = cfg or StreamOVConfig()
    torch.manual_seed(21)
    t = 12
    sv = torch.rand(t)
    sa = torch.rand(t)
    scob = torch.rand(t) * 0.5
    sqv = torch.rand(t)
    sqa = torch.rand(t)

    ev, ea, eav, base = build_multimodal_evidence(sv, sa, scob, sqv, sqa)
    bank = MemoryBank()
    bank = update_long_short_memory(bank, base, ev, ea, eav, window_offset=0, cfg=cfg)
    bank = update_long_short_memory(bank, base, ev, ea, eav, window_offset=t, cfg=cfg)

    trigger = ResponseTrigger(dim=cfg.hidden_dim)
    h = torch.randn(2, 1, cfg.hidden_dim)  # prefilling-only h_{t,0}
    logits = trigger(h)
    target = torch.tensor([1, 0])
    loss = F.cross_entropy(logits, target)

    return {
        "evidence_max": float(base.max()),
        "memory_size": float(len(bank)),
        "trigger_loss": float(loss.detach()),
        "trigger_params": float(sum(p.numel() for p in trigger.parameters()) / 1e6),
    }


def evaluation_demo(cfg: StreamOVConfig | None = None) -> dict[str, Any]:
    """Smoke: paper table ordering and design claims."""
    cfg = cfg or StreamOVConfig()
    step = training_step_demo(cfg)
    t1 = table_sovbench()
    t2 = table_streamingbench()
    t4 = table_ablation_components()
    t6 = table_trigger_ablation()

    stream_methods = ["Qwen3-Omni-30B", "ROMA-7B", "StreamOV"]
    qa_avgs = {m: t1[m]["avg_qa"] for m in stream_methods}
    f1_online = {m: t1[m]["f1"] for m in ("ROMA-7B", "StreamOV") if t1[m]["f1"] is not None}

    return {
        **step,
        "streamov_best_sovbench_avg": t1["StreamOV"]["avg_qa"] == max(qa_avgs.values()),
        "beats_qwen3_omni_avg": t1["StreamOV"]["avg_qa"] > t1["Qwen3-Omni-30B"]["avg_qa"],
        "beats_roma_avg": t1["StreamOV"]["avg_qa"] > t1["ROMA-7B"]["avg_qa"],
        "best_trigger_f1": t6["StreamOV + Trigger (h0)"]["f1"] == max(v["f1"] for v in t6.values()),
        "long_memory_helps": t4["full_with_long_memory"] > t4["qa_plus_qag"],
        "beats_streamingbench_av": t2["StreamOV"]["av_avg"] > t2["Qwen3-Omni-30B"]["av_avg"],
        "frame_budget_64": float(cfg.frame_budget_sovbench) == 64.0,
        "trigger_under_20m": float(cfg.trigger_params_m) < 20.0,
        "sovbench_f1_gap": float(t1["StreamOV"]["f1"] - f1_online["ROMA-7B"]),
    }
