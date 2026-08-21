"""GOPEX integration: neural UV atlas + generated mesh assets."""

from __future__ import annotations

from typing import Any


def gopex_stub_links() -> dict[str, str]:
    return {
        "amara_spatial": "AmaraSpatial-10K generated meshes (arXiv:2604.23018)",
        "light_harmony3d": "3DGS mesh insertion — UV validity for texture baking",
        "cinematte": "Matting + texture transfer needs foldover-free UVs",
        "p2gs": "Exposure-invariant 3DGS — mesh export UV cleanup",
        "control_room_17": "Teleplay asset prep — atlas construction for props",
        "penpot": "Design stack — checkerboard UV validation previews",
    }


def ltx_plan_stub() -> dict[str, Any]:
    return {
        "phase": "validation_first_uv_atlas",
        "pipeline": [
            "mesh_normalize_repair_chart_split",
            "disk_like_validation_routing",
            "lbo_spectral_features_siren_uv",
            "tutte_warmup_c2_symmetric_dirichlet_barrier",
            "flip_reject_retry_or_fallback_pca",
            "area_scaled_shelf_pack",
            "atlas_checkerboard_validation",
        ],
        "representation": "untrained SIREN continuous reparameterization",
        "prior": "LBO spectral features + Tutte residual warm-up",
        "scope": "fixed-chart validity; recutting optional via OptCuts/BFF",
    }


def ltx_prep_notes() -> list[str]:
    return [
        "Normalize mesh to unit box; cotangent LBO with lumped mass (k=16 modes).",
        "Route non-disk generated fragments to PCA/per-face fallback before pack.",
        "Short validated pass: 750–1000 Adam iters (~6 s compact) for zero-flip maps.",
        "Use libigl SLIM/BFF when recutting allowed; neural solver for supplied-chart repair.",
        "Large-scale Rust atlas path validates routing/packing without full PyTorch per chart.",
    ]


def integration_bundle() -> dict[str, Any]:
    return {
        "gopex_stubs": gopex_stub_links(),
        "ltx_plan": ltx_plan_stub(),
        "prep_notes": ltx_prep_notes(),
    }
