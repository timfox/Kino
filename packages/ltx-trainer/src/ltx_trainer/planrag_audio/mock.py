"""PlanRAG-Audio smoke: toy plan, fusion, and SQL sketch."""

from __future__ import annotations

from typing import Any

from ltx_trainer.planrag_audio.config import PlanRagAudioConfig
from ltx_trainer.planrag_audio.fusion import TimeSegment, fuse_nearest_midpoint
from ltx_trainer.planrag_audio.sql_toy import merged_sql_sketch


def example_plan_dict() -> dict[str, Any]:
    """Minimal Θ(q) plan for tests and evaluation_demo."""
    return {
        "streams": ["transcription", "speaker"],
        "filters": {"speaker": "SPEAKER_02", "transcription": "keyword"},
        "fusion": {"anchor": "transcript"},
        "output": {"return_fields": ["start", "end", "text"]},
    }


def evaluation_smoke(cfg: PlanRagAudioConfig | None = None) -> dict[str, Any]:
    c = cfg or PlanRagAudioConfig()
    plan = example_plan_dict()
    base = [TimeSegment(20.5, 22.1, "tx")]
    targets = [TimeSegment(20.0, 22.5, "sp1"), TimeSegment(100.0, 101.0, "far")]
    fused = fuse_nearest_midpoint(base, targets, tau=c.fusion_tau_seconds)
    sql = merged_sql_sketch(plan)
    ratio = c.gemini_planrag_tokens_k / c.gemini_full_tokens_k
    return {
        "paper": c.paper_arxiv,
        "fused_payload": fused[0][1].payload if fused[0][1] is not None else None,
        "gemini_token_ratio": ratio,
        "sql_lines": len(sql.splitlines()),
        "fusion_tau_s": c.fusion_tau_seconds,
    }
