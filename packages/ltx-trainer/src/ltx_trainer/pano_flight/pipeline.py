"""Survey demos and lightweight classifiers."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.pano_flight.config import PanoFlightConfig
from ltx_trainer.pano_flight.gaps import gap_severity_map, gaps_card
from ltx_trainer.pano_flight.projections import projection_catalogue
from ltx_trainer.pano_flight.strategies import classify_method, strategies_card
from ltx_trainer.pano_flight.taxonomy import TaskPillar, taxonomy_card


def evaluation_demo_run(cfg: PanoFlightConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PanoFlightConfig()
    maps = gap_severity_map(cfg.height, cfg.width)
    u = torch.tensor(float(cfg.width // 2))
    v = torch.tensor(float(cfg.height // 2))
    from ltx_trainer.pano_flight.projections import erp_pixel_to_spherical, spherical_to_unit

    phi, theta = erp_pixel_to_spherical(u, v, cfg.width, cfg.height)
    unit = spherical_to_unit(phi, theta)
    return {
        "erp_shape": [cfg.height, cfg.width],
        "gap_map_keys": list(maps.keys()),
        "equator_unit_norm": float(unit.norm()),
        "classify_osrt": classify_method("OSRT").value,
        "classify_bifuse": classify_method("BiFuse").value,
        "pillars": [p.value for p in TaskPillar],
    }


def stitching_pipeline_card() -> dict[str, Any]:
    return {
        "stages": [
            "data_preprocessing",
            "data_association",
            "geometric_alignment",
            "image_blending",
        ],
        "association_types": ["spatial", "geometric", "photometric"],
        "blending": ["linear", "multiband", "poisson"],
    }


def cross_method_demo() -> dict[str, Any]:
    return {
        "gaps": gaps_card(),
        "strategies": strategies_card(),
        "projections": projection_catalogue(),
    }


def cross_task_demo() -> dict[str, Any]:
    return taxonomy_card()
