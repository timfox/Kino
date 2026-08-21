"""Sphere360 evaluation smoke — computed parse + LTX plan + optional ERP tensor."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sphere360._bootstrap import ensure_repo_root
from ltx_trainer.sphere360.benchmarks import PAPER_HUB, SPHERE360_CLIP_DURATION_S
from ltx_trainer.sphere360.pipeline import evaluation_demo, pipeline_demo


def evaluation_smoke(*, clip_key: str = "dQw4w9WgXcQ_042") -> dict[str, Any]:
    ensure_repo_root()
    from gopex_datasets.sphere360 import parse_clip_id
    from gopex_datasets.sphere360.knowledge import sphere360_knowledge_blob

    ref = parse_clip_id(clip_key)
    kb = sphere360_knowledge_blob()
    pipe = pipeline_demo()
    ev = evaluation_demo()
    out: dict[str, Any] = {
        "package": "sphere360",
        "hub": PAPER_HUB,
        "clip_id": ref.clip_id,
        "segment_index": ref.segment_index,
        "clip_duration_s": SPHERE360_CLIP_DURATION_S,
        "train_clips": kb["scale"]["train_clips"],
        "plan_phases": pipe["plan_phases"],
        "eval_plan_phases": ev["plan_phases"],
    }

    try:
        import torch

        from ltx_trainer.pano360.geometry import is_equirectangular_size
        from ltx_trainer.pano360.synthetic import synthesize_equirect_rgb

        rgb = synthesize_equirect_rgb(64, 128, seed=1)
        erp_ok = is_equirectangular_size(128, 64)
        # LogC3 proxy on normalized linear ERP slice
        lin = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0 * 2.0
        from ltx_trainer.hdr_ingest import linear_scene_to_logc3_display

        logc3 = linear_scene_to_logc3_display(lin.unsqueeze(0)).squeeze(0)
        out.update(
            {
                "torch": True,
                "erp_valid": erp_ok,
                "logc3_mean": round(float(logc3.mean()), 4),
                "logc3_finite": bool(torch.isfinite(logc3).all()),
            }
        )
    except ImportError:
        out["torch"] = False

    return out
