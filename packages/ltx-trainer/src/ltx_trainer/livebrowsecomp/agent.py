"""RedSearcher-style search-agent scaffold (§4.1)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ltx_trainer.livebrowsecomp.trajectory import RetrievalHit, SearchRound, SearchTrajectory

DEFAULT_SYSTEM_PROMPT = """You are a deep search assistant. Your primary role is to perform rigorous, multi-step,
multi-source investigations on any topic. For each user request, actively seek out and
cross-check information from credible sources, then integrate findings into a comprehensive,
accurate, and objective response.

When you have collected sufficient information and are ready to deliver the definitive
response, wrap the entire final answer in <answer></answer> tags.
"""

FORCE_ANSWER_PROMPT = """You have reached the maximum number of research steps or context limit.
Based on all the information you have gathered so far, provide your final answer.
Put your final answer between <answer> and </answer>.
"""

TOOL_NAMES = ("search", "visit", "code_sandbox")

_SEARCH_RE = re.compile(
    r'search\s*\(\s*["\'](.+?)["\']\s*\)',
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ScaffoldConfig:
    max_steps: int = 250
    max_context_tokens: int = 256_000
    temperature: float = 0.7
    top_p: float = 0.9
    search_top_k: int = 10


@dataclass
class ScaffoldState:
    question: str
    question_id: str
    step: int = 0
    reasoning_log: str = ""
    trajectory: SearchTrajectory = field(default_factory=lambda: SearchTrajectory(question_id=""))

    def __post_init__(self) -> None:
        if not self.trajectory.question_id:
            self.trajectory.question_id = self.question_id


SearchFn = Callable[[str], list[RetrievalHit]]
VisitFn = Callable[[str, str], str]


def parse_search_calls(assistant_text: str) -> list[str]:
    return [m.group(1).strip() for m in _SEARCH_RE.finditer(assistant_text)]


def run_scaffold_loop(
    question: str,
    *,
    question_id: str = "0",
    search_fn: SearchFn | None = None,
    max_rounds: int = 5,
    cfg: ScaffoldConfig | None = None,
) -> SearchTrajectory:
    """
    Record a search trajectory without calling an external LLM.

    ``search_fn`` returns retrieval hits for each parsed ``search(...)`` call.
    """
    cfg = cfg or ScaffoldConfig()
    state = ScaffoldState(question=question, question_id=question_id)
    state.trajectory.question_id = question_id

    def _default_search(q: str) -> list[RetrievalHit]:
        return [RetrievalHit(doc_id="stub", snippet=f"Snippet for: {q[:80]}")]

    search_fn = search_fn or _default_search

    reasoning = f"Hypothesis-driven investigation for: {question[:120]}"
    for r in range(max_rounds):
        queries = parse_search_calls(reasoning) or [question[:80]]
        hits: list[RetrievalHit] = []
        for q in queries[:1]:
            hits.extend(search_fn(q))
        rnd = SearchRound(
            round_idx=r,
            query=queries[0],
            reasoning_before=reasoning,
            hits=hits,
            reasoning_after=reasoning + "\nIntegrated retrieval.",
        )
        state.trajectory.rounds.append(rnd)
        state.step += 1
        if state.step >= cfg.max_steps:
            break
        reasoning += f"\nFollow-up search round {r + 1}."

    state.trajectory.final_answer = f"<answer>stub</answer>"
    return state.trajectory


def scaffold_spec(cfg: ScaffoldConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ScaffoldConfig()
    return {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "force_answer_prompt": FORCE_ANSWER_PROMPT,
        "tools": list(TOOL_NAMES),
        "hyperparameters": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        },
        "limits": {
            "max_steps": cfg.max_steps,
            "max_context_tokens": cfg.max_context_tokens,
            "search_top_k": cfg.search_top_k,
        },
        "search_backend": "serper.dev",
        "visit_backend": "jina",
    }
