"""End-to-end OMAF multirate pipeline stub (Fig. 3)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.oma_fme.analysis_reuse import simulate_default_erp_times, simulate_ladder_times
from ltx_trainer.oma_fme.config import OmaFmeConfig
from ltx_trainer.oma_fme.ladder import ladder_representation_count, plan_for_strategy
from ltx_trainer.oma_fme.projections import cmp_tile_count, projection_metadata, stub_erp_to_cmp_faces
from ltx_trainer.oma_fme.variants import FRAMEWORK_VARIANTS, ProjectionFormat, SharingStrategy, variant_name


def parse_variant(name: str) -> tuple[ProjectionFormat, SharingStrategy]:
    parts = name.upper().split("-")
    if len(parts) != 2:
        raise ValueError(f"unknown variant {name!r}")
    return ProjectionFormat(parts[0]), SharingStrategy(parts[1])


def run_pipeline(
    config: OmaFmeConfig | None = None,
    *,
    erp_batch: Any = None,
) -> dict[str, Any]:
    """Plan ladder, optional CMP split, encode-time estimate, OMAF packaging metadata."""
    cfg = config or OmaFmeConfig()
    proj, strat = parse_variant(cfg.variant)
    jobs = plan_for_strategy(strat)
    tiles = cmp_tile_count(proj)
    times = simulate_ladder_times(strat, proj)
    ref_times = simulate_default_erp_times()

    if erp_batch is not None:
        import torch

        if isinstance(erp_batch, torch.Tensor):
            faces = stub_erp_to_cmp_faces(erp_batch) if proj == ProjectionFormat.CMP else {}
        else:
            faces = {}
    else:
        faces = {}

    return {
        "variant": cfg.variant,
        "projection": projection_metadata(proj),
        "strategy": strat.value,
        "anchor_quality": cfg.anchor_quality,
        "encode_jobs_per_tile": len(jobs),
        "total_encodes": ladder_representation_count(projection_tiles=tiles, strategy=strat),
        "times_s": times,
        "reference_times_s": ref_times,
        "cmp_faces": list(faces.keys()) if faces else [],
        "omaf_dash": {
            "packaged": True,
            "projection_signalling": True,
            "region_based": proj == ProjectionFormat.CMP,
        },
        "framework_variants": FRAMEWORK_VARIANTS,
        "canonical_name": variant_name(proj, strat),
    }
