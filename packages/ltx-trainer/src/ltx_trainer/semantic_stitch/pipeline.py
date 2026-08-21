"""SemanticStitch end-to-end CPU pipeline."""

from __future__ import annotations

from dataclasses import asdict

from ltx_trainer.semantic_stitch.config import GFLOPS, PARAMS_M, SemanticStitchConfig
from ltx_trainer.semantic_stitch.integration import proceduralsky_card, proceduralsky_pipeline_plan
from ltx_trainer.semantic_stitch.losses import composite_coverage_loss
from ltx_trainer.semantic_stitch.masks import optimize_seam_masks, saliency_union
from ltx_trainer.semantic_stitch.metrics import metric_bundle
from ltx_trainer.semantic_stitch.seam import graph_cut_seam, stitch_images, voronoi_seam
from ltx_trainer.semantic_stitch.synthetic import synthetic_overlap_pair


def run_stitch_smoke(cfg: SemanticStitchConfig | None = None, *, seed: int = 0) -> dict:
    cfg = cfg or SemanticStitchConfig(image_size=64, max_mask_epochs=20)
    data = synthetic_overlap_pair(size=cfg.image_size, seed=seed)
    obj = saliency_union(data["m1"], data["m2"])
    l_gc, _ = graph_cut_seam(data["gradient"])
    l_v, _ = voronoi_seam(data["gradient"])
    l1, l2, hist = optimize_seam_masks(obj, cfg=cfg, seed=seed)
    stitched_gc = stitch_images(data["i1"], data["i2"], l_gc, 1.0 - l_gc)
    stitched_ours = stitch_images(data["i1"], data["i2"], l1, l2)
    losses = composite_coverage_loss(obj, l1, l2, cfg)
    m_gc = metric_bundle(stitched_gc, l_gc, obj)
    m_ours = metric_bundle(stitched_ours, l1, obj)
    return {
        "object_pixels": float(obj.sum()),
        "losses": losses,
        "optimization_steps": len(hist),
        "metrics_gc": m_gc,
        "metrics_ours": m_ours,
        "stitched_shape": list(stitched_ours.shape),
    }


def run_proceduralsky_bridge(*, skies_project: str = "sphere360-hdr-timelapse") -> dict:
    return {
        "card": proceduralsky_card(),
        "pipeline": proceduralsky_pipeline_plan(skies_project=skies_project),
    }


def evaluation_smoke(*, seed: int = 0, skies_project: str = "sphere360-hdr-timelapse") -> dict:
    cfg = SemanticStitchConfig(image_size=64, max_mask_epochs=15)
    return {
        "paper": "SemanticStitch",
        "arxiv": "2511.12084",
        "stitch": run_stitch_smoke(cfg, seed=seed),
        "proceduralsky": run_proceduralsky_bridge(skies_project=skies_project),
        "efficiency": {"params_M": PARAMS_M, "GFLOPs": GFLOPS},
        "config": asdict(cfg),
    }
