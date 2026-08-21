"""Alignment paradigm taxonomy (Fig. 1)."""

from __future__ import annotations

from enum import Enum


class AlignmentParadigm(str, Enum):
    LATENT_QUERY = "latent_query_alignment"
    TEXTUAL_READOUT = "textual_localization_readout"
    INTERPRETABLE_SAE = "interpretable_sparse_concepts"


def paradigm_properties() -> dict[str, dict[str, str | bool]]:
    return {
        AlignmentParadigm.LATENT_QUERY.value: {
            "end_to_end": True,
            "interpretable": False,
            "spatial_grounding": "opaque_latent_queries",
            "examples": "LISA, RAS, PixelLM",
        },
        AlignmentParadigm.TEXTUAL_READOUT.value: {
            "end_to_end": False,
            "interpretable": False,
            "spatial_grounding": "discrete_boxes_or_patch_indices",
            "examples": "Seg-Zero, Text4Seg, VisionReasoner",
        },
        AlignmentParadigm.INTERPRETABLE_SAE.value: {
            "end_to_end": True,
            "interpretable": True,
            "spatial_grounding": "multi_slot_heatmap_from_sparse_concepts",
            "examples": "SegCompass",
        },
    }
