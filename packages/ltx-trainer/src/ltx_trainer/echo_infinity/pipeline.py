"""Echo-Infinity framework card, paper tables, training/inference demos (arXiv:2606.04527)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.echo_infinity.config import EchoInfinityConfig
from ltx_trainer.echo_infinity.dmd import dmd_step_loss
from ltx_trainer.echo_infinity.inference import rollout_chunks
from ltx_trainer.echo_infinity.kv_cache import constant_memory_budget_tokens
from ltx_trainer.echo_infinity.ltx_plan import ltx_integration_plan
from ltx_trainer.echo_infinity.memory import MemoryQueryStack, memory_token_count
from ltx_trainer.echo_infinity.relative_rope import advance_mature_rope, layout_for_step, verify_layout_in_range


def framework_card(cfg: EchoInfinityConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EchoInfinityConfig()
    return {
        "name": "Echo-Infinity",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_page,
        "code": cfg.code_url,
        "base_model": cfg.base_model,
        "mechanisms": [
            "End-to-end Memory Queries (filter / abstract / compress evicted KV)",
            "Unified Relative RoPE (train + inference within [0, fmax])",
            "Three-tier KV: sink + local window + memory queries",
            "Two-stage DMD: 5s standard → 60s streaming long-tune",
        ],
        "memory_budget_tokens": constant_memory_budget_tokens(
            cfg.num_sink_frames,
            cfg.local_window_frames,
            cfg.num_memory_query_frames,
            cfg.chunk_size_frames,
            cfg.tokens_per_frame,
        ),
        "throughput_fps": cfg.throughput_fps,
        "throughput_overhead_pct": cfg.throughput_overhead_pct,
        "infinite_demo": ">1.3M frames / 24h @ 18.5 FPS (paper claim, H100)",
    }


def table_long_vbench_30_240() -> list[dict[str, Any]]:
    """Table 1 — VBench-Long 30s / 240s + user preference (%)."""
    return [
        {
            "model": "LongLive",
            "params": "1.3B",
            "fps": 20.7,
            "quality_30s": 83.59,
            "semantic_30s": 80.28,
            "pref_30s_pct": 10.47,
            "quality_240s": 79.79,
            "pref_240s_pct": 6.13,
        },
        {
            "model": "MemFlow",
            "params": "1.3B",
            "fps": 18.7,
            "quality_30s": 83.35,
            "semantic_30s": 80.85,
            "pref_30s_pct": 10.13,
            "quality_240s": 79.31,
            "pref_240s_pct": 5.93,
        },
        {
            "model": "Memorize-and-Generate",
            "params": "1.3B",
            "fps": 21.7,
            "quality_30s": 83.69,
            "semantic_30s": 81.01,
            "pref_30s_pct": 14.73,
            "quality_240s": 75.49,
            "pref_240s_pct": 2.13,
        },
        {
            "model": "∞-RoPE",
            "params": "1.3B",
            "fps": 17.0,
            "quality_30s": 83.38,
            "semantic_30s": 74.67,
            "pref_30s_pct": 5.13,
            "quality_240s": 79.99,
            "pref_240s_pct": 14.13,
        },
        {
            "model": "Echo-Infinity",
            "params": "1.3B",
            "fps": 18.5,
            "quality_30s": 85.61,
            "semantic_30s": 82.01,
            "pref_30s_pct": 59.53,
            "quality_240s": 81.23,
            "pref_240s_pct": 71.67,
        },
    ]


def table_interactive_60s() -> list[dict[str, Any]]:
    """Table 2 — MemFlow interactive 60s benchmark."""
    return [
        {"model": "LongLive", "quality": 79.38, "clip_0_10": 34.08, "clip_50_60": 30.49},
        {"model": "MemFlow", "quality": 79.91, "clip_0_10": 33.48, "clip_50_60": 30.23},
        {"model": "Memorize-and-Generate", "quality": 79.15, "clip_0_10": 33.58, "clip_50_60": 30.27},
        {"model": "∞-RoPE", "quality": 79.22, "clip_0_10": 33.15, "clip_50_60": 30.17},
        {"model": "Echo-Infinity", "quality": 81.71, "clip_0_10": 34.10, "clip_50_60": 30.74},
    ]


def table_short_vbench_5s() -> list[dict[str, Any]]:
    """Table 3 — VBench 5s (selected rows)."""
    return [
        {"model": "Wan-2.1", "fps": 0.78, "total": 84.26, "quality": 85.30, "semantic": 80.09},
        {"model": "LongLive", "fps": 20.7, "total": 83.29, "quality": 84.09, "semantic": 80.06},
        {"model": "Echo-Infinity (w/o Memory Update)", "fps": 18.9, "total": 84.57, "quality": 85.51, "semantic": 80.80},
        {"model": "Echo-Infinity (w/ Memory Update)", "fps": 18.5, "total": 85.35, "quality": 86.32, "semantic": 81.49},
    ]


def table_ablation_240s() -> list[dict[str, Any]]:
    """Table 4 / Tab. 5 — 240s MovieGen ablations."""
    return [
        {
            "variant": "(a) w/o Memory Queries",
            "subject_consistency": 96.15,
            "background_consistency": 95.27,
            "dynamic_degree": 64.78,
            "aesthetic_quality": 58.60,
            "clip_score": 32.78,
        },
        {
            "variant": "(b) w/o Unified Relative RoPE",
            "subject_consistency": 96.34,
            "background_consistency": 95.81,
            "dynamic_degree": 64.05,
            "aesthetic_quality": 59.83,
            "clip_score": 33.12,
        },
        {
            "variant": "(c) w/ Self Forcing ODE init",
            "subject_consistency": 96.85,
            "background_consistency": 96.12,
            "dynamic_degree": 52.04,
            "aesthetic_quality": 58.49,
            "clip_score": 34.07,
        },
        {
            "variant": "Echo-Infinity",
            "subject_consistency": 96.58,
            "background_consistency": 95.93,
            "dynamic_degree": 68.61,
            "aesthetic_quality": 58.67,
            "clip_score": 34.19,
        },
    ]


def table_ablation_extended() -> list[dict[str, Any]]:
    """Appendix Tab. 5 — NQ and gate ablations."""
    return [
        {"variant": "NQ=1", "subject_consistency": 94.72, "dynamic_degree": 67.39},
        {"variant": "NQ=5", "subject_consistency": 96.44, "dynamic_degree": 69.47},
        {"variant": "w/o Gate", "subject_consistency": 95.31, "dynamic_degree": 68.43},
        {"variant": "Echo-Infinity (NQ=3)", "subject_consistency": 96.58, "dynamic_degree": 68.61},
    ]


def training_step_demo(cfg: EchoInfinityConfig | None = None) -> dict[str, float]:
    """Smoke: memory update + DMD mismatch + short AR rollout."""
    cfg = cfg or EchoInfinityConfig()
    torch.manual_seed(27)
    device = "cpu"
    batch, dim = 2, cfg.hidden_dim
    tpf = cfg.tokens_per_frame
    evict_tokens = cfg.chunk_size_frames * tpf

    memory = MemoryQueryStack(cfg)
    q = memory.init_queries(batch)
    k_evict = torch.randn(batch, evict_tokens, dim)
    v_evict = torch.randn(batch, evict_tokens, dim)
    q_new = memory.update(q, k_evict, v_evict)

    x_t = torch.randn(batch, dim)
    mu_r = torch.randn(batch, dim)
    mu_f = torch.randn(batch, dim)
    t = torch.tensor([0.4, 0.5])
    dmd_loss = dmd_step_loss(x_t, mu_r, mu_f, t)

    roll = rollout_chunks(4, cfg, batch=batch, device=device)
    layout = layout_for_step(f_star=100, cfg=cfg, has_memory=True)
    mature = advance_mature_rope(layout, num_sink=cfg.num_sink_frames, fmax=cfg.fmax)

    return {
        "memory_query_delta": float((q_new - q).detach().pow(2).mean()),
        "dmd_loss": float(dmd_loss),
        "rollout_loss_tail": float(roll["loss_tail"]),
        "rope_in_range": float(verify_layout_in_range(layout, cfg.fmax)),
        "mature_rope_in_range": float(verify_layout_in_range(mature, cfg.fmax)),
        "memory_tokens": float(memory_token_count(cfg)),
    }


def evaluation_demo(cfg: EchoInfinityConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EchoInfinityConfig()
    ours_30 = next(r for r in table_long_vbench_30_240() if r["model"] == "Echo-Infinity")
    ours_240 = ours_30
    return {
        "framework": framework_card(cfg),
        "ltx_plan": ltx_integration_plan(cfg),
        "training": training_step_demo(cfg),
        "rollout_16_chunks": rollout_chunks(16, cfg),
        "paper_tables": {
            "long_vbench": table_long_vbench_30_240(),
            "interactive_60s": table_interactive_60s(),
            "short_5s": table_short_vbench_5s(),
            "ablation_240s": table_ablation_240s(),
            "ablation_extended": table_ablation_extended(),
        },
        "delta_quality_30s": ours_30["quality_30s"] - 83.69,
        "delta_pref_240s_pct": ours_240["pref_240s_pct"] - 14.13,
    }
