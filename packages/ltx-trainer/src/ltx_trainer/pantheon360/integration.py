"""proceduralsky.com + digital-twin stack integrations (arXiv:2605.25449)."""

from __future__ import annotations

from typing import Any

PROCEDURALSKY_URL = "https://proceduralsky.com"
DEFAULT_SKIES_PROJECT = "sphere360-hdr-timelapse"
DEFAULT_ERP_BUCKETS = "1024x512x41"


def proceduralsky_card() -> dict[str, Any]:
    """
    How Pantheon360 trajectory-controlled 360° video pairs with proceduralsky HDR skies.

    Procedural Skies trains a sidecar LoRA on 2:1 ERP HDR timelapse (``kino-procedural-skies-360.sh``).
    Pantheon360 supplies geometry-consistent scene motion and novel views for digital-twin worlds.
    """
    return {
        "partner": PROCEDURALSKY_URL,
        "role": "illumination_and_environment",
        "pantheon360_role": (
            "3D Cache → V_geo ERP scaffold + SVD diffusion for exact camera trajectories "
            "(Street View interpolation, stabilization, sparse-view synthesis)"
        ),
        "workflow": [
            "Capture or source sparse 360° ERP anchors (e.g. sphere360 HDR timelapse, Street View).",
            "Reconstruct 3D Cache (PI3/VGGT); render V_geo along C_target.",
            "Run Pantheon360 diffusion for photorealistic, temporally consistent 360° video.",
            "Train or apply proceduralsky HDR sky LoRA on matching 1024×512 ERP buckets.",
            "Optional: merge multi-capture overlaps with SemanticStitch (foreground-aware seams) before prep.",
            "Composite: scene video from Pantheon360 + proceduralsky environment map for CG/VR/IBL.",
        ],
        "gopex_scripts": {
            "skies_lora": "./scripts/kino-procedural-skies-360.sh",
            "pantheon360": "./scripts/kino-pantheon360.sh",
            "semantic_stitch": "./scripts/kino-semantic-stitch.sh",
        },
        "default_skies_project": DEFAULT_SKIES_PROJECT,
        "erp_buckets": DEFAULT_ERP_BUCKETS,
        "notes": (
            "GOPEX stub does not ship SVD weights or PI3; fold hook annotates latents with "
            "cache_fusion_readiness for joint 360° scene + sky training. Full Pantheon360 release "
            "required for production trajectory control."
        ),
    }


def proceduralsky_pipeline_plan(
    *,
    skies_project: str = DEFAULT_SKIES_PROJECT,
    work_root: str | None = None,
) -> dict[str, Any]:
    """Ordered steps linking proceduralsky HDR prep to Pantheon360-style 360° generation."""
    wr = work_root or "${WORK_ROOT}"
    pre = f"{wr}/precomputed/{skies_project}"
    merged = f"{wr}/precomputed/merged_{skies_project}"
    return {
        "skies_project": skies_project,
        "work_root": wr,
        "phases": [
            {
                "id": "skies_media",
                "command": f"GOPEX_SKIES_PROJECT={skies_project} ./scripts/kino-procedural-skies-360.sh probe",
                "artifact": f"{pre}/ltx_manifest/pano360_probe.jsonl",
            },
            {
                "id": "semantic_stitch",
                "command": f"GOPEX_SKIES_PROJECT={skies_project} ./scripts/kino-procedural-skies-360.sh semantic-stitch",
                "artifact": f"{pre}/stitched_erp/",
                "optional": True,
            },
            {
                "id": "skies_prep",
                "command": f"GOPEX_SKIES_PROJECT={skies_project} ./scripts/kino-procedural-skies-360.sh prep",
                "artifact": merged,
            },
            {
                "id": "pantheon360_smoke",
                "command": "./scripts/kino-pantheon360.sh smoke",
                "artifact": "V_geo + latent concat + dual-anchor fusion demo",
            },
            {
                "id": "fold_annotate",
                "command": "Annotate 360° latent shards with pantheon360.cache_fusion_readiness (fold_registry)",
                "artifact": "meta.pantheon360 on ERP training clips",
            },
            {
                "id": "skies_train",
                "command": f"GOPEX_SKIES_PROJECT={skies_project} ./scripts/kino-procedural-skies-360.sh train",
                "artifact": f"{wr}/runs/procedural_skies_360_hdr_lora",
            },
        ],
        "digital_twin": [
            "Static HDR environment: proceduralsky LoRA / catalog EXR",
            "Dynamic scene motion: Pantheon360 along user trajectory",
            "Joint metric targets: Tables 1–2 (FVD, MET3R) + proceduralsky HDR audit",
        ],
    }
