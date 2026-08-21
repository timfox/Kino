"""Fine-grained category-aware routing (§2.3)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.visa.config import VisaConfig
from ltx_trainer.visa.taxonomy import RoutingStrategy, lookup_category


def route_by_strategy(
    strategy: RoutingStrategy,
    *,
    qwen_answer: str,
    step_answer: str,
    vlm_answer: str,
    llm_selected: str,
) -> dict[str, Any]:
    if strategy == RoutingStrategy.VLM_SPECTRAL:
        return {"selected": vlm_answer, "router": "vlm_spectral"}
    if strategy == RoutingStrategy.DIRECT_QWEN:
        return {"selected": qwen_answer, "router": "direct_qwen"}
    if strategy == RoutingStrategy.DIRECT_STEP:
        return {"selected": step_answer, "router": "direct_step"}
    return {"selected": llm_selected, "router": "llm_judge"}


def disagree_then_route(
    *,
    category_id: str,
    qwen_answer: str,
    step_answer: str,
    vlm_answer: str | None = None,
    llm_selected: str | None = None,
    cfg: VisaConfig | None = None,
) -> dict[str, Any]:
    """Resolve model disagreement using category-aware rules."""
    cfg = cfg or VisaConfig()
    cat = lookup_category(category_id)
    strategy = cat["strategy"]
    vlm_answer = vlm_answer or qwen_answer
    llm_selected = llm_selected or step_answer
    routed = route_by_strategy(
        strategy,
        qwen_answer=qwen_answer,
        step_answer=step_answer,
        vlm_answer=vlm_answer,
        llm_selected=llm_selected,
    )
    return {
        "category_id": category_id,
        "category_name": cat["name"],
        "strategy": strategy.value,
        "qwen_answer": qwen_answer,
        "step_answer": step_answer,
        "final_answer": routed["selected"],
        "router": routed["router"],
        "llm_backbone": cfg.llm_backbone,
    }


def routing_demo(seed: int = 0) -> dict[str, Any]:
    """Toy routing across representative categories."""
    cases = [
        ("duration_estimation", "C", "B", "C"),
        ("env_scene", "A", "B", "A"),
        ("event_counting", "B", "A", "B"),
        ("emotion_intention", "D", "C", "D"),
    ]
    results = []
    for i, (cat_id, q, s, vlm) in enumerate(cases):
        results.append(
            disagree_then_route(
                category_id=cat_id,
                qwen_answer=q,
                step_answer=s,
                vlm_answer=vlm,
                llm_selected=q,
            )
        )
    return {"num_cases": len(results), "routes": results, "seed": seed}
