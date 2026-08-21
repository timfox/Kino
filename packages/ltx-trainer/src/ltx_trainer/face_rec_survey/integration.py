"""Links to GOPEX face datasets and LTX character consistency."""

from __future__ import annotations

from typing import Any


def gopex_dataset_links() -> dict[str, Any]:
    return {
        "face_age_10k": {
            "hub": "prithivMLmods/Face-Age-10K",
            "agent_doc": "data/face_age_10k/AGENTS.md",
            "role": "Age-band consistency for character / cast reference QA",
        },
        "finevideo": {
            "hub": "HuggingFaceFV/finevideo",
            "role": "Rich video metadata; face-heavy scenes for cast consistency",
        },
        "lfw": {
            "hub": "http://vis-www.cs.umass.edu/lfw/",
            "role": "Uncontrolled still-face benchmark cited in survey Table 1",
        },
        "incantation_elden": {
            "hub": "MatrixTeam/incantation-elden-ring-scenes",
            "role": "Game cast / boss face consistency captions",
        },
    }


def ltx_cast_consistency_notes() -> list[str]:
    return [
        "Use Face-Age-10K bands when locking character age across LTX teleplay shots.",
        "Illumination and pose confounders (§6) map to HDR ERP and multi-view 360° prep.",
        "Partial occlusion (glasses, hands) aligns with CR17 cast_refs anchor QA.",
    ]
