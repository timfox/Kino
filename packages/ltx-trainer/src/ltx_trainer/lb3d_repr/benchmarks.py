"""Table 1 (explicit repr.) and Table 2 (3DGS optimization) from CGF survey."""

from __future__ import annotations

from typing import Any

TABLE1_EXPLICIT: list[dict[str, Any]] = [
    {
        "representation": "Polygonal Meshes",
        "continuity": "medium",
        "connectivity": "high",
        "topology": "medium",
        "strengths": "Standard rendering; explicit vertex density; efficient rasterization",
        "weaknesses": "Irregular graphs for DL; non-differentiable topology changes",
        "key_refs": ["MeshCNN [HHF*19]", "Pixel2Mesh [WZL*18]"],
    },
    {
        "representation": "Point Clouds",
        "continuity": "low",
        "connectivity": "low",
        "topology": "high",
        "strengths": "Native sensor output; easy resize; no connectivity overhead",
        "weaknesses": "No surface definition; radius tuning; weak flat regions",
        "key_refs": ["PointNet [QSMG17]", "Pulsar [LZ21]"],
    },
    {
        "representation": "Neural Surfaces",
        "continuity": "high",
        "connectivity": "high",
        "topology": "low",
        "strengths": "Infinite resolution; differentiable; regular UV domain for CNNs",
        "weaknesses": "Fixed/low genus without multiple charts",
        "key_refs": ["AtlasNet [GFK*18b]", "FoldingNet [YFST18]"],
    },
]

TABLE2_3DGS_OPTIMIZATION: list[dict[str, Any]] = [
    {
        "goal": "Pruning",
        "memory": "significant",
        "geometry": "neutral",
        "artifacts": "neutral",
        "mechanism": "Redundant / invisible Gaussian culling",
        "key_refs": ["PKK*24", "LRS*24"],
    },
    {
        "goal": "Compression",
        "memory": "significant",
        "geometry": "neutral",
        "artifacts": "neutral",
        "mechanism": "VQ / hash-grids for SH and attributes",
        "key_refs": ["NSW24", "MBHE24"],
    },
    {
        "goal": "Anti-Aliasing",
        "memory": "neutral",
        "geometry": "neutral",
        "artifacts": "significant",
        "mechanism": "3D smoothing / mip filters",
        "key_refs": ["YCH*24", "YLCL24"],
    },
    {
        "goal": "Artifact Removal",
        "memory": "neutral",
        "geometry": "neutral",
        "artifacts": "significant",
        "mechanism": "Per-pixel sort / ray-tracing (popping fix)",
        "key_refs": ["RSP*24", "YSG24"],
    },
    {
        "goal": "Geometry Alignment",
        "memory": "neutral",
        "geometry": "significant",
        "artifacts": "neutral",
        "mechanism": "SDF / depth / normal priors",
        "key_refs": ["GL24", "YLX*24", "LZB*24"],
    },
]

# Paper-cited NeRF / 3DGS anchors (Sec. 4.4, 4.5)
PAPER_ANCHORS: dict[str, Any] = {
    "nerf_volume_rendering": "Eq. (3) transmittance integral via stratified quadrature [MST*20]",
    "instant_ngp_speedup": "Real-time hash-grid NeRF [MESK22]",
    "3dgs_pipeline": "SfM init → tile sort → alpha blend [KKLD23]",
    "efficientnerf_fps": ">200 fps inference with NerfTree [HLC*22]",
    "plenoctrees_speedup": ">3000× vs vanilla NeRF [YLT*21]",
}


def table_explicit_representations() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE1_EXPLICIT]


def table_3dgs_optimization() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE2_3DGS_OPTIMIZATION]


def paper_anchors() -> dict[str, Any]:
    return dict(PAPER_ANCHORS)
