"""proceduralsky.com + 360° ERP pipeline integration."""

from __future__ import annotations

from typing import Any

PROCEDURALSKY_URL = "https://proceduralsky.com"
DEFAULT_SKIES_PROJECT = "sphere360-hdr-timelapse"


def proceduralsky_card() -> dict[str, Any]:
    """
    Foreground-aware seam carving for multi-capture ERP sky panoramas.

    Procedural sky timelapses often merge bracketed or adjacent captures into 2:1 ERP
    buckets; SemanticStitch keeps salient sun/cloud structure intact at seams.
    """
    return {
        "partner": PROCEDURALSKY_URL,
        "role": "erp_overlap_seam_carving",
        "semantic_stitch_role": (
            "Object-aware soft seams when fusing overlapping equirectangular sky captures "
            "before HDR preprocess / LoRA training"
        ),
        "workflow": [
            "Align overlapping ERP or partial-sky captures (homography / UDIS-style warp stub).",
            "Detect salient regions: sun disk, bright cloud mass, horizon landmarks.",
            "Run Composite Coverage Loss seam optimization; avoid cuts through salient sky objects.",
            "Blend with S = L1·I1 + L2·I2; feed stitched ERP into kino-procedural-skies-360.sh prep.",
        ],
        "gopex_scripts": {
            "skies_lora": "./scripts/kino-procedural-skies-360.sh",
            "semantic_stitch": "./scripts/kino-semantic-stitch.sh",
        },
        "notes": (
            "GOPEX stub uses numpy saliency proxies; production path wires SelfReformer masks "
            "and FastViT seam head from the paper release."
        ),
    }


def proceduralsky_pipeline_plan(
    *,
    skies_project: str = DEFAULT_SKIES_PROJECT,
    work_root: str | None = None,
) -> dict[str, Any]:
    """Insert SemanticStitch between media probe and split-caption when multi-capture merge is needed."""
    wr = work_root or "${WORK_ROOT}"
    return {
        "skies_project": skies_project,
        "work_root": wr,
        "phases": [
            {
                "id": "probe",
                "command": f"GOPEX_SKIES_PROJECT={skies_project} ./scripts/kino-procedural-skies-360.sh probe",
            },
            {
                "id": "semantic_stitch",
                "command": f"GOPEX_SKIES_PROJECT={skies_project} ./scripts/kino-procedural-skies-360.sh semantic-stitch",
                "artifact": f"{wr}/precomputed/{skies_project}/stitched_erp/",
                "when": "Multiple captures per scene or bracket overlap need salient-aware seams",
            },
            {
                "id": "split_caption",
                "command": f"GOPEX_SKIES_PROJECT={skies_project} ./scripts/kino-procedural-skies-360.sh split-caption",
            },
            {
                "id": "prep_train",
                "command": f"GOPEX_SKIES_PROJECT={skies_project} ./scripts/kino-procedural-skies-360.sh prep && train",
            },
        ],
        "saliency_targets": [
            "Sun disk and corona bloom",
            "Dominant cloud formations",
            "Horizon line continuity (exclude lower neck/collar analog: terrain)",
        ],
    }
