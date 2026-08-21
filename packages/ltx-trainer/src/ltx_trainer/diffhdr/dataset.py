"""Polyhaven HDRI curation pipeline metadata (Sec. 3.1)."""

from __future__ import annotations

POLYHAVEN_URL = "https://polyhaven.com/"

MOTION_PATTERNS = (
    "highlight_zoom_in_out",
    "shadow_zoom_in_out",
    "rotation_120deg_x3",
)

DATASET_STATS = {
    "hdri_sources": 800,
    "sequences": 5400,
    "frames_per_sequence": 81,
    "color_space": "linear Rec.709",
    "focal_length_mm_range": [(18, 30), (50, 70)],
    "rotation_deg_per_segment": 120,
}

EVAL_DATASETS = {
    "si_hdr": {"type": "image", "metrics": ["hdr_vdp3", "pu21_piqe", "fid"]},
    "cinematic_video": {"type": "video", "metrics": ["fovvideovdp", "dover", "musiq", "clipiqa"]},
    "polyhaven_synthetic": {"type": "video", "held_out": 50},
    "in_the_wild": {"type": "video", "source": "Pexels", "count": 50},
    "veo2": {"type": "video", "count": 10},
}


def dataset_summary() -> dict[str, object]:
    return {
        "source": POLYHAVEN_URL,
        "motion_patterns": list(MOTION_PATTERNS),
        "stats": DATASET_STATS,
        "eval_datasets": EVAL_DATASETS,
        "backbone": "Wan-2.1-VACE-14B (stub)",
        "vae": "Wan-2.1-VAE 4×8×8 (stub)",
    }
