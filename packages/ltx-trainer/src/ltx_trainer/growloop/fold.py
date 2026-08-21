"""AV-fold sidecar for GrowLoop conversation evaluation stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.growloop.config import GrowLoopConfig
from ltx_trainer.growloop.heuristic import cascaded_final_score


def growloop_meta_block() -> dict[str, Any]:
    cfg = GrowLoopConfig()
    return {
        "growloop": {
            "paper": cfg.paper_arxiv,
            "org": cfg.org,
            "case_count": cfg.case_count,
            "rubric_dimensions": cfg.quality_dimensions,
            "inter_annotator_agreement": cfg.inter_annotator_agreement,
            "merged_agreement_gemini": cfg.merged_agreement_gemini,
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach GrowLoop cascaded score proxy to fold sidecar."""
    out = dict(data)
    out.update(growloop_meta_block())
    duration_s = float(data.get("duration_s", 0.0))
    fatal_issue = bool(data.get("fatal_issue", False))
    quality_raw = float(data.get("quality_raw", 3.5))
    final = cascaded_final_score(fatal_issue, quality_raw)
    out["growloop"].update(
        {
            "duration_s": duration_s,
            "quality_raw": quality_raw,
            "fatal_issue": fatal_issue,
            "final_score_bucket": final,
        }
    )
    return out
