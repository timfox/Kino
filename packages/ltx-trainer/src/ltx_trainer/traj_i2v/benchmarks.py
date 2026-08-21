"""Reference metrics from Bompai et al. Table III (arXiv:2605.16420)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.traj_i2v.config import (
    DEFAULT_FPS,
    DEFAULT_NUM_GENERATED_FRAMES,
    DEFAULT_POST_HEIGHT,
    DEFAULT_POST_WIDTH,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    SG_I2V_REF,
)

# Table III — quantitative comparison (lower is better for all reported metrics)
TABLE3_QUANTITATIVE: dict[str, dict[str, float | str]] = {
    "sg_i2v": {
        "lpips": 0.149,
        "temporal_smoothness_px_per_frame": 1.14,
        "brisque": 25.52,
        "trajectory_error_px": 9.31,
    },
    "optical_flow": {
        "lpips": 0.110,
        "temporal_smoothness_px_per_frame": 0.39,
        "brisque": 17.47,
        "trajectory_error_px": 27.71,
    },
    "rife": {
        "lpips": 0.130,
        "temporal_smoothness_px_per_frame": 0.59,
        "brisque": 42.46,
        "trajectory_error_px": 34.75,
    },
    "ground_truth": {
        "lpips": "ref",
        "temporal_smoothness_px_per_frame": 1.42,
        "brisque": 23.64,
        "trajectory_error_px": 28.70,
    },
}

TABLE1_RESOLUTION: dict[str, str | int | float] = {
    "original": "3840x2160 @ 30 FPS",
    "post_processing": f"{DEFAULT_POST_WIDTH}x{DEFAULT_POST_HEIGHT} @ {DEFAULT_FPS} FPS",
    "generated_frames": DEFAULT_NUM_GENERATED_FRAMES,
}

DATASET_ASV = {
    "vessel_ids": {99999: "green", 100000: "yellow"},
    "sog_kn_range": {99999: (0.0, 3.5), 100000: (0.0, 4.2)},
    "log_columns": [
        "timestamp_utc",
        "id",
        "longitude",
        "latitude",
        "sog_kn",
        "cog_deg",
        "heading_deg",
        "mission_phase_uuid",
    ],
    "temporal_alignment": "t_log = t_video + 21 s (manual, approximate)",
    "scale_px_per_m_clip": 28.3,
}

METHODS = {
    "sg_i2v": "Self-guided trajectory I2V on SVD backbone (Namekata et al., arXiv:2411.04989)",
    "rife": "Practical-RIFE bidirectional intermediate flow (Huang et al., ECCV 2022)",
    "optical_flow": "Farneback dense flow warp from reference frame (OpenCV)",
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "sg_i2v_ref": SG_I2V_REF,
        "table3_quantitative": TABLE3_QUANTITATIVE,
        "table1_resolution": TABLE1_RESOLUTION,
        "dataset_asv": DATASET_ASV,
        "methods": METHODS,
    }
