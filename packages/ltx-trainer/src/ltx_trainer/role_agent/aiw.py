"""Agent-In-World: failure memory, reflection parsing, curriculum resampling."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ltx_trainer.role_agent.failure_modes import FAILURE_MODES_BY_DOMAIN
from ltx_trainer.role_agent.prompts import TASK_RETRIEVAL_TEMPLATE


@dataclass
class FailureReflection:
    dominant_type: str
    core_lesson: str
    retrieval_query: str
    task_id: str
    domain: str
    detail: str = ""
    critical_step: str = ""


@dataclass
class FailureMemory:
    """Offline library M of failure modes and associated tasks."""

    entries: list[FailureReflection] = field(default_factory=list)

    def add(self, reflection: FailureReflection) -> None:
        self.entries.append(reflection)

    def unique_modes(self, domain: str | None = None) -> list[str]:
        modes = {e.dominant_type for e in self.entries if domain is None or e.domain == domain}
        return sorted(modes)

    def retrieve(self, query: str, *, domain: str | None = None, top_k: int = 5) -> list[FailureReflection]:
        q_tokens = set(query.lower().split())
        scored: list[tuple[float, FailureReflection]] = []
        for entry in self.entries:
            if domain is not None and entry.domain != domain:
                continue
            hay = f"{entry.dominant_type} {entry.core_lesson} {entry.retrieval_query}".lower()
            overlap = len(q_tokens & set(hay.split()))
            if overlap:
                scored.append((overlap / max(len(q_tokens), 1), entry))
        scored.sort(key=lambda x: -x[0])
        seen: set[str] = set()
        out: list[FailureReflection] = []
        for _, entry in scored:
            if entry.task_id in seen:
                continue
            seen.add(entry.task_id)
            out.append(entry)
            if len(out) >= top_k:
                break
        return out


_REFLECTION_RE = re.compile(
    r"<reflection>\s*"
    r"DOMINANT_TYPE:\s*(?P<type>[^\n]+)\s*"
    r"(?:DETAIL:\s*(?P<detail>[^\n]+)\s*)?"
    r"(?:CRITICAL_STEP:\s*(?P<critical>[^\n]+)\s*)?"
    r"CORE_LESSON:\s*(?P<lesson>[^\n]+)\s*"
    r"RETRIEVAL_QUERY:\s*(?P<query>[^\n<]+)\s*"
    r"</reflection>",
    re.IGNORECASE | re.DOTALL,
)

_SELECTED_TASKS_RE = re.compile(
    r"<selected_tasks>\s*(?P<body>.*?)\s*</selected_tasks>",
    re.IGNORECASE | re.DOTALL,
)
_INDEX_LINE_RE = re.compile(
    r"INDEX(?:/TASK/REFLECTIONS)?:\s*(?P<idx>\d+)",
    re.IGNORECASE,
)


def parse_reflection(text: str, *, task_id: str, domain: str) -> FailureReflection | None:
    m = _REFLECTION_RE.search(text)
    if not m:
        return None
    return FailureReflection(
        dominant_type=m.group("type").strip(),
        core_lesson=m.group("lesson").strip(),
        retrieval_query=m.group("query").strip(),
        task_id=task_id,
        domain=domain,
        detail=(m.group("detail") or "").strip(),
        critical_step=(m.group("critical") or "").strip(),
    )


def parse_selected_tasks(text: str) -> list[int]:
    """Parse Figure 9 <selected_tasks> indices."""
    m = _SELECTED_TASKS_RE.search(text)
    if not m:
        return []
    body = m.group("body")
    indices: list[int] = []
    for line in body.splitlines():
        hit = _INDEX_LINE_RE.search(line)
        if hit:
            indices.append(int(hit.group("idx")))
    return indices


def format_task_retrieval_prompt(reflection: FailureReflection, candidates: Sequence[str]) -> str:
    mode_library = reflection.dominant_type
    return TASK_RETRIEVAL_TEMPLATE.format(
        error_pattern=f"{reflection.dominant_type}: {reflection.core_lesson}",
        candidates_text="\n".join(f"[{i}] {c}" for i, c in enumerate(candidates)),
        mode_library=mode_library,
    )


def classify_failure_heuristic(trajectory_text: str, domain: str) -> str:
    """Rule-based fallback when no LLM reflection is available."""
    text = trajectory_text.lower()
    modes = FAILURE_MODES_BY_DOMAIN.get(domain, [])
    if "search[" in text and text.count("search[") > 3:
        for m in ("repeated_retrieval_query", "repeated_query", "excessive_browsing"):
            if m in modes:
                return m
    if "format" in text or "invalid action" in text:
        return "action_format_error"
    if "loop" in text or text.count("go to") > 4:
        for m in ("navigation_loop", "repetitive_exploration"):
            if m in modes:
                return m
    if "give up" in text or "nothing else" in text:
        for m in ("premature_give_up", "premature_termination", "premature_answer"):
            if m in modes:
                return m
    return modes[0] if modes else "unknown_failure"


def curriculum_resample_weights(
    task_ids: Sequence[str],
    memory: FailureMemory,
    *,
    boost: float = 2.0,
) -> dict[str, float]:
    """Reshape p_D by up-weighting tasks linked to stored failure modes."""
    weights = {tid: 1.0 for tid in task_ids}
    retrieved_tasks = {e.task_id for e in memory.entries}
    for tid in task_ids:
        if tid in retrieved_tasks:
            weights[tid] = boost
    return weights
