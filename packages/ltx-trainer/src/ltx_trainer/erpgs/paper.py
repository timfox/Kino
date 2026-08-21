"""Framework card for agents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.erpgs.config import (
    LAMBDA_DN,
    LAMBDA_F,
    LAMBDA_S,
    OPT_ITERATIONS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    REG_START_ITER,
)


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "components": [
            "Omnidirectional 3D Gaussian splatting (ERP projection + Jacobian)",
            "Geometric regularization: rendered normal vs depth-normal (DNE)",
            "Scale regularization L_s and flattening L_f",
            "Distortion-aware pixel weight W (cos latitude)",
            "Viewpoint-dependent mask M_p for rig/tripod obstacles",
        ],
        "optimization": {
            "iterations": OPT_ITERATIONS,
            "reg_start": REG_START_ITER,
            "lambda_dn": LAMBDA_DN,
            "lambda_f": LAMBDA_F,
            "lambda_s": LAMBDA_S,
        },
        "baselines": ["EgoNeRF", "ODGS", "OmniGS"],
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.erpgs.pipeline import evaluation_demo_run

    out = evaluation_demo_run()
    return {"package": "erpgs", **out}
