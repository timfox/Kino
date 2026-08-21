"""GOPEX 3D representation stubs linked from the CGF survey."""

from __future__ import annotations

from typing import Any

GOPEX_3D_STUBS: dict[str, dict[str, str]] = {
    "erpgs": {
        "doc": "documents/ERPGS.md",
        "module": "ltx_trainer.erpgs",
        "role": "ERP 3D Gaussian splatting (panoramic explicit-volumetric hybrid)",
    },
    "tpgs": {
        "doc": "documents/TPGS.md",
        "module": "ltx_trainer.tpgs",
        "role": "Transition-plane cubemap 3DGS panoramas",
    },
    "p2gs": {
        "doc": "documents/P2GS.md",
        "module": "ltx_trainer.p2gs",
        "role": "Exposure-invariant HDR 3DGS",
    },
    "r5dgs": {
        "doc": "documents/R5DGS.md",
        "module": "ltx_trainer.r5dgs",
        "role": "Semantic 4DGS + rigid extrapolation (dynamic Sec. 6)",
    },
    "frng": {
        "doc": "documents/FRNG.md",
        "module": "ltx_trainer.frng",
        "role": "Relightable neural Gaussians (material-aware Sec. 5.2)",
    },
    "pantheon360": {
        "doc": "documents/PANTHEON360.md",
        "module": "ltx_trainer.pantheon360",
        "role": "3D-aware 360° video diffusion",
    },
    "gimbal360": {
        "doc": "documents/GIMBAL360.md",
        "module": "ltx_trainer.gimbal360",
        "role": "Panoramic completion / large-scale ERP",
    },
    "sphere360": {
        "doc": "data/sphere360/AGENTS.md",
        "module": "gopex_datasets.sphere360",
        "role": "360° video + FOA audio (real-world unbounded data)",
    },
    "deblur_nvs": {
        "doc": "documents/DEBLUR_NVS.md",
        "module": "ltx_trainer.deblur_nvs",
        "role": "Sparse-view NVS (generalization Sec. 5.3)",
    },
    "traj_i2v": {
        "doc": "documents/TRAJ_I2V.md",
        "module": "ltx_trainer.traj_i2v",
        "role": "GPS-guided I2V reconstruction",
    },
    "latenthdr": {
        "doc": "documents/LATENTHDR.md",
        "module": "ltx_trainer.latenthdr",
        "role": "HDR volumetric / appearance fields",
    },
    "physthdr_gs": {
        "doc": "documents/PHYSTHDR_GS.md",
        "module": "ltx_trainer.physthdr_gs",
        "role": "Physically inspired HDR-NVS (Sec. 8 physical plausibility)",
    },
    "mirage": {
        "doc": "documents/MIRAGE.md",
        "module": "ltx_trainer.mirage",
        "role": "Latent spatial memory cache for video world models (dynamic Sec. 6)",
    },
    "phyworld": {
        "doc": "documents/PHYWORLD.md",
        "module": "ltx_trainer.phyworld",
        "role": "Physics-faithful flow world model (temporal plausibility Sec. 6)",
    },
}


def gopex_stub_links() -> dict[str, dict[str, str]]:
    return {k: dict(v) for k, v in GOPEX_3D_STUBS.items()}


def ltx_nvs_pipeline_notes() -> list[str]:
    return [
        "SfM point clouds (COLMAP) initialize 3DGS [KKLD23] — shared with kino prep pipelines.",
        "NeRF-style latent training uses AV-fold merged_native; see LTX_FOLD_HOOKS.md.",
        "Panoramic ERP workflows: ErpGS + Sphere360 catalog for unbounded real-world Sec. 5.1.",
        "Dynamic scenes: R5DGS rigid constraints complement warping / space-time Sec. 6.",
        "Video world models: Mirage latent cache + PhyWorld physics proxy on merged_native.",
        "Shape–radiance ambiguity: prefer mesh export (Marching Cubes / 2DGS) for metrology hooks.",
    ]


def survey_to_gopex_mapping() -> list[dict[str, Any]]:
    """Map survey sections to local stubs."""
    return [
        {"survey_section": "3.1 meshes", "gopex": ["erpgs", "tpgs"]},
        {"survey_section": "3.1.2 points", "gopex": ["r5dgs", "p2gs"]},
        {"survey_section": "4.3 NeRF", "gopex": ["latenthdr", "deblur_nvs"]},
        {"survey_section": "4.5 3DGS", "gopex": ["erpgs", "tpgs", "p2gs", "r5dgs"]},
        {"survey_section": "5.1 unbounded", "gopex": ["sphere360", "gimbal360", "pantheon360"]},
        {"survey_section": "5.2 materials", "gopex": ["frng", "physthdr_gs"]},
        {"survey_section": "6 dynamic", "gopex": ["r5dgs", "pantheon360", "mirage", "phyworld"]},
        {"survey_section": "7 robotics SLAM", "gopex": ["r5dgs", "deblur_nvs", "mirage"]},
    ]
