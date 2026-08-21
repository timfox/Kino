"""Framework summary."""

from __future__ import annotations

from typing import Any

from ltx_trainer.prior_flow.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, REFINE_ITERATIONS


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "code": CODE_URL,
        "components": {
            "DCCL": "dual-cost collaborative lookup on primitive + orthogonal correlation pyramids",
            "ODDC": "ortho-driven distortion compensation with warp&GW confidence (Eq. 13–16)",
            "orthogonal_view": "90° x-axis spherical rotation → low-distortion polar prior (Fig. 2–3)",
        },
        "variants": ("PriOr-RAFT", "PriOr-GMA", "PriOr-SKFlow"),
        "refine_iterations": REFINE_ITERATIONS,
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.prior_flow.pipeline import evaluation_demo_run

    return {"package": "prior_flow", **evaluation_demo_run()}
