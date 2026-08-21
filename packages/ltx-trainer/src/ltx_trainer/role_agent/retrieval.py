"""AIW task retrieval for curriculum reshaping (Figure 9, Algorithm 1 steps 21–22)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ltx_trainer.role_agent.aiw import FailureMemory, FailureReflection


def format_candidate_entry(idx: int, reflection: FailureReflection, *, task_prompt: str = "") -> str:
    prompt = task_prompt or reflection.task_id
    return (
        f"TASK: {prompt}\n"
        f"RETRIEVED_REFLECTION: {reflection.dominant_type} — {reflection.core_lesson}"
    )


def build_candidate_library(
    memory: FailureMemory,
    task_prompts: dict[str, str],
) -> list[tuple[int, str, str]]:
    """Unique (index, task_id, formatted candidate) from failure memory."""
    seen: set[str] = set()
    out: list[tuple[int, str, str]] = []
    for entry in memory.entries:
        if entry.task_id in seen:
            continue
        seen.add(entry.task_id)
        idx = len(out)
        prompt = task_prompts.get(entry.task_id, entry.task_id)
        out.append((idx, entry.task_id, format_candidate_entry(idx, entry, task_prompt=prompt)))
    return out


def retrieve_task_ids_by_overlap(
    reflection: FailureReflection,
    memory: FailureMemory,
    *,
    top_k: int = 3,
) -> list[str]:
    hits = memory.retrieve(reflection.retrieval_query or reflection.dominant_type, domain=reflection.domain, top_k=top_k)
    return [h.task_id for h in hits if h.task_id != reflection.task_id]


def retrieve_task_ids_llm(
    reflection: FailureReflection,
    candidates: Sequence[str],
    llm_client,
) -> list[int]:
    from ltx_trainer.role_agent.llm_backend import llm_retrieve_task_indices

    if not candidates or llm_client is None:
        return []
    try:
        return llm_retrieve_task_indices(llm_client, reflection, list(candidates))
    except Exception:
        return []


def curriculum_resample_weights_retrieval(
    task_ids: Sequence[str],
    memory: FailureMemory,
    task_prompts: dict[str, str],
    *,
    boost: float = 2.0,
    llm_client=None,
    use_llm_retrieval: bool = False,
) -> dict[str, float]:
    """Reshape p_D by failure-mode retrieval (token overlap + optional LLM Figure 9)."""
    weights = {tid: 1.0 for tid in task_ids}
    if not memory.entries:
        return weights

    library = build_candidate_library(memory, task_prompts)
    candidate_texts = [c[2] for c in library]
    index_to_task = {c[0]: c[1] for c in library}

    boosted: set[str] = set()
    for entry in memory.entries[-8:]:
        if entry.task_id in weights:
            boosted.add(entry.task_id)
        for tid in retrieve_task_ids_by_overlap(entry, memory, top_k=5):
            if tid in weights:
                boosted.add(tid)
        if use_llm_retrieval and llm_client is not None and candidate_texts:
            for idx in retrieve_task_ids_llm(entry, candidate_texts, llm_client):
                tid = index_to_task.get(idx)
                if tid and tid in weights:
                    boosted.add(tid)

    for tid in boosted:
        weights[tid] = boost
    return weights


def retrieval_demo(memory: FailureMemory, task_prompts: dict[str, str]) -> dict[str, Any]:
    """CPU smoke: show overlap retrieval for latest failure."""
    if not memory.entries:
        return {"ok": False, "reason": "empty memory"}
    latest = memory.entries[-1]
    lib = build_candidate_library(memory, task_prompts)
    return {
        "ok": True,
        "latest_mode": latest.dominant_type,
        "overlap_hits": retrieve_task_ids_by_overlap(latest, memory),
        "candidate_count": len(lib),
    }
