"""LTX / Kino integration plan for Echo-Infinity AR long-video hooks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.echo_infinity.config import EchoInfinityConfig


def ltx_integration_plan(cfg: EchoInfinityConfig | None = None) -> dict[str, Any]:
    """How to wire Echo-Infinity into Gopex causal LTX rollouts."""
    cfg = cfg or EchoInfinityConfig()
    return {
        "paper": cfg.paper_arxiv,
        "target_backbone": "Wan2.1-T2V-1.3B class → Gopex LTX-2 causal AV fold path",
        "training_stages": [
            {
                "name": "stage1_standard",
                "duration_s": 5,
                "data": "VidProM-style prompts",
                "objective": "Causal-forcing DMD + memory queries + unified relative RoPE",
                "iterations": cfg.stage1_iterations,
            },
            {
                "name": "stage2_streaming",
                "duration_s": 60,
                "data": "single prompt switch (LongLive protocol)",
                "objective": "streaming long tuning; detach Q across 5s sub-clip boundaries",
                "iterations": cfg.stage2_iterations,
            },
        ],
        "inference_hooks": [
            "Store per-layer K/V pre-RoPE; apply relative RoPE ids before each attention.",
            "Inject MemoryQueryStack KQ/VQ after sink, before local window.",
            "On local-window eviction: update Q from last-layer evicted K/V (Eq. 4).",
            "Use GOPEX_LTX3_MINUTE_COMPOSE for segment boundaries; carry last-frame + Q state.",
        ],
        "env_knobs": {
            "GOPEX_ECHO_INFINITY_MEMORY": "1 — enable memory-query path",
            "GOPEX_ECHO_INFINITY_ROPE": "1 — unified relative RoPE (default on)",
            "GOPEX_ECHO_INFINITY_NS": str(cfg.num_sink_frames),
            "GOPEX_ECHO_INFINITY_NW": str(cfg.local_window_frames),
            "GOPEX_ECHO_INFINITY_NQ": str(cfg.num_memory_query_frames),
            "GOPEX_ECHO_INFINITY_FMAX": str(cfg.fmax),
        },
        "merge_target": "merged_native precomputed catalog after caption+prep",
        "eval_protocol": [
            "VBench-Long 30s/240s tables (Tab. 1)",
            "MemFlow 60s interactive (Tab. 2)",
            "VBench 5s short (Tab. 3)",
        ],
    }
