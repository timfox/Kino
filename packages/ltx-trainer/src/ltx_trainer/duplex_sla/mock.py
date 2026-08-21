"""DuplexSLA action spill + bench smoke (arXiv:2605.20755)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.duplex_sla.action_queue import format_tool_call, spill_action_tokens
from ltx_trainer.duplex_sla.config import DuplexSlaConfig


def evaluation_smoke(cfg: DuplexSlaConfig | None = None) -> dict[str, Any]:
    c = cfg or DuplexSlaConfig()
    tokens = ["search", "music", "play"]
    chunks = spill_action_tokens(tokens, max_per_chunk=1)
    marker = format_tool_call("search_music", "relaxing music")
    return {
        "paper": c.paper_arxiv,
        "n_action_chunks": len(chunks),
        "bench_total": c.bench_total_cases,
        "chunk_ms": c.chunk_ms,
        "tool_marker_len": len(marker),
    }
