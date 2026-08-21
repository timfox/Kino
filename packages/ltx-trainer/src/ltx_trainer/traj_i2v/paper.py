"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.traj_i2v.benchmarks import PAPER_TITLE, benchmarks_bundle
from ltx_trainer.traj_i2v.config import PAPER_ARXIV, PAPER_URL, SG_I2V_REF


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    t3 = b["table3_quantitative"]
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "sg_i2v_ref": SG_I2V_REF,
        "problem": (
            "Reconstruct dropped frames in top-down maritime drone video where vessels are "
            "small and sea texture is low; GPS logs provide motion cues when pixels alone are ambiguous."
        ),
        "method": {
            "gps_to_pixel": "Equirectangular EN + yaw rotation + click-calibrated px/m scale (Eq. 1–5)",
            "conditioning": "SG-I2V: 2 vessel boxes + 4 static corner anchors, N=14 frames",
            "generation": "Pre-trained SG-I2V on SVD (no domain fine-tuning)",
            "baselines": "Farneback optical-flow extrapolation, Practical-RIFE interpolation",
        },
        "results_table3": {
            "sg_i2v_brisque": t3["sg_i2v"]["brisque"],
            "sg_i2v_temporal": t3["sg_i2v"]["temporal_smoothness_px_per_frame"],
            "sg_i2v_traj_px": t3["sg_i2v"]["trajectory_error_px"],
            "gt_brisque": t3["ground_truth"]["brisque"],
        },
        "reference_metrics": b,
        "integration": (
            "GOPEX implements GPS→pixel mapping, SG-I2V conditioning export, and metric stubs; "
            "wire external SG-I2V weights for production inference. Complements equirectangular "
            "HDR environment maps (e.g. proceduralsky.com) for CG sky/sea lighting."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.traj_i2v.mock import evaluation_smoke

    return evaluation_smoke()
