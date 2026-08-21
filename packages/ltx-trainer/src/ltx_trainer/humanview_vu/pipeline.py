"""Survey demos and method routing."""

from __future__ import annotations

from typing import Any

from ltx_trainer.humanview_vu.formulation import grpo_group_normalize, run_formulation_demo, sft_loss_proxy
from ltx_trainer.humanview_vu.taxonomy import classify_method, taxonomy_card


def evaluation_demo_run(num_frames: int = 16, query: str | None = None) -> dict[str, Any]:
    demo = run_formulation_demo(num_frames=num_frames, query=query)
    samples = [
        classify_method("TimeChat"),
        classify_method("MovieChat"),
        classify_method("Video-R1"),
        classify_method("Video-o3"),
        classify_method("StreamingVLM"),
    ]
    rewards = [0.2, 0.5, 0.8, 1.0]
    return {
        **demo,
        "method_routing": samples,
        "sft_loss_proxy": sft_loss_proxy([-1.2, -0.8, -0.3]),
        "grpo_normalized_rewards": grpo_group_normalize(rewards),
        "taxonomy_leaves": len(taxonomy_card()["watch"]) + len(taxonomy_card()["remember"]) + len(taxonomy_card()["reason"]),
    }
