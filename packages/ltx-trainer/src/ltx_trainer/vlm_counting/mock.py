"""CPU smoke entry for validate_paper_stubs."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke() -> dict[str, Any]:
    pipeline = load_sibling(__file__, "pipeline")
    return {
        "framework": pipeline.framework_card(),
        "evaluation": pipeline.evaluation_demo(),
        "training_step": pipeline.training_step_demo(),
    }
