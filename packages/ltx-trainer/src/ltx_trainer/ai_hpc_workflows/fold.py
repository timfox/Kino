"""AV-fold sidecar: AI+HPC workflow design hints on captions."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ai_hpc_workflows.constants import PAPER_ARXIV
from ltx_trainer.ai_hpc_workflows.orchestration import recommend_orchestrator
from ltx_trainer.ai_hpc_workflows.phases import map_task_to_resource


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption")
        or meta.get("prompt")
        or meta.get("text")
        or data.get("caption")
        or ""
    ).lower()
    adaptivity = (
        "dynamic"
        if any(w in caption for w in ("adaptive", "feedback", "loop", "spawn"))
        else "iterative"
    )
    task_hint = "train" if "train" in caption else ("simulate" if "simulation" in caption else "infer")
    orch = recommend_orchestrator("dynamic" if adaptivity == "dynamic" else "iterative")
    out["ai_hpc_workflows"] = {
        "arxiv_id": PAPER_ARXIV,
        "adaptivity": adaptivity,
        "orchestrator_hint": orch["primary"],
        "resource_map": map_task_to_resource(task_hint),
        "tip_focus": "workflow_engine" if adaptivity == "dynamic" else "separate_phases",
    }
    return out
