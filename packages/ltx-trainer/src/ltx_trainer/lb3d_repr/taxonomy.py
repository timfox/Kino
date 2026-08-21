"""Survey taxonomy (Fig. 1–2) and unified formulation Eq. (1)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lb3d_repr.config import DomainNature, RepresentationClass

# f_θ : D → S,  ξ = f_θ(γ(q))
UNIFIED_FORMULATION = {
    "mapping": "f_θ : D → S",
    "output": "ξ = f_θ(γ(q))",
    "domain_examples": ["3D coordinates s", "view direction d", "time t", "primitive indices"],
    "output_examples": ["occupancy", "SDF", "radiance", "SH coefficients", "vertex positions"],
}


def survey_sections() -> list[dict[str, str]]:
    return [
        {"id": "2", "title": "Taxonomy and problem statement", "focus": "Eq. (1), explicit vs implicit"},
        {"id": "3", "title": "Explicit representations", "focus": "Meshes, points, neural surfaces / atlas"},
        {"id": "4", "title": "Volumetric representations", "focus": "RBF, NeRF, acceleration, 3DGS"},
        {"id": "5", "title": "Real-world scenes", "focus": "Unbounded, materials, generalization"},
        {"id": "6", "title": "Dynamic scenes", "focus": "Warping fields, space-time fields"},
        {"id": "7", "title": "Applications", "focus": "Digital humans, robotics, generative content"},
        {"id": "8", "title": "Future research", "focus": "Hybrid primitives, LRMs, shape–radiance ambiguity"},
    ]


def taxonomy_tree() -> dict[str, Any]:
    """Hierarchical overview matching Fig. 1."""
    return {
        "explicit": {
            "discrete": ["polygonal_meshes", "point_clouds"],
            "continuous": ["neural_surfaces", "atlas_net", "folding_net"],
        },
        "volumetric": {
            "discrete": ["voxel_grids", "octrees"],
            "continuous": ["neural_fields", "nerf", "primitive_splatting_3dgs"],
        },
        "extensions": {
            "real_world": ["unbounded_warping", "reflectance_fields", "generalizable_mvs"],
            "dynamic": ["warping_fields", "space_time_functions", "feature_grid_4d"],
        },
    }


def landscape_nodes() -> list[dict[str, Any]]:
    """Fig. 2 — fidelity vs deployability with class and domain."""
    return [
        {
            "name": "point_clouds",
            "class": RepresentationClass.SURFACE.value,
            "domain": DomainNature.DISCRETE.value,
            "memory": "low",
            "fidelity": "medium",
            "deployability": "high",
        },
        {
            "name": "polygonal_meshes",
            "class": RepresentationClass.SURFACE.value,
            "domain": DomainNature.DISCRETE.value,
            "memory": "medium",
            "fidelity": "high",
            "deployability": "high",
        },
        {
            "name": "neural_surfaces",
            "class": RepresentationClass.SURFACE.value,
            "domain": DomainNature.CONTINUOUS.value,
            "memory": "low",
            "fidelity": "medium",
            "deployability": "medium",
        },
        {
            "name": "voxel_grids",
            "class": RepresentationClass.VOLUMETRIC.value,
            "domain": DomainNature.DISCRETE.value,
            "memory": "high",
            "fidelity": "medium",
            "deployability": "medium",
        },
        {
            "name": "nerf_mlp",
            "class": RepresentationClass.VOLUMETRIC.value,
            "domain": DomainNature.CONTINUOUS.value,
            "memory": "low",
            "fidelity": "high",
            "deployability": "low",
        },
        {
            "name": "hash_grid_nerf",
            "class": RepresentationClass.HYBRID.value,
            "domain": DomainNature.CONTINUOUS.value,
            "memory": "medium",
            "fidelity": "high",
            "deployability": "medium",
        },
        {
            "name": "3d_gaussian_splatting",
            "class": RepresentationClass.VOLUMETRIC.value,
            "domain": DomainNature.DISCRETE.value,
            "memory": "high",
            "fidelity": "high",
            "deployability": "high",
        },
    ]


def acceleration_strategies() -> list[dict[str, str]]:
    """Sec. 4.4 — five NeRF acceleration families."""
    return [
        {"id": "subdivision", "summary": "Many tiny MLPs / cells (KiloNeRF, DeRF)"},
        {"id": "surface_localization", "summary": "Skip empty space via sparse grids, SDF, depth oracles"},
        {"id": "baking", "summary": "Precompute diffuse / SH in sparse structures (PlenOctrees)"},
        {"id": "efficient_integration", "summary": "AutoInt, deterministic segment decoders"},
        {"id": "feature_grids", "summary": "Tri-planes, hash grids, TensoRF factorization"},
    ]


def future_directions() -> list[str]:
    return [
        "Hybrid splattable neural primitives (memory vs speed trade-off)",
        "Amortized inference via LRMs, DUSt3R, MASt3R, VGGT",
        "Robust pose-free / sparse-view pipelines",
        "Resolving shape–radiance ambiguity for geometry-critical tasks",
        "Physical plausibility: PBR, dynamics, simulation coupling",
    ]
