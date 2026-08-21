"""Search-augmented QA environment (Figure 7 prompt shape, Table 2 benchmarks)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.role_agent.e5_retriever import E5Retriever
from ltx_trainer.role_agent.types import TaskSpec

_SEARCH_RE = re.compile(r"^search\[(.+)\]$", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^answer\[(.+)\]$", re.IGNORECASE)

# Minimal in-memory corpus for local CPU / replay-adjacent training (NQ/Hotpot-style)
SEARCH_QA_CORPUS: dict[str, list[str]] = {
    "france capital": ["Paris is the capital and largest city of France."],
    "france": ["France is a country in Western Europe."],
    "paris": ["Paris is the capital of France."],
    "python creator": ["Python was created by Guido van Rossum."],
    "guido van rossum": ["Guido van Rossum created Python in 1991."],
    "bridge river thames": ["Tower Bridge crosses the River Thames in London."],
    "tower bridge": ["Tower Bridge is a combined bascule and suspension bridge in London."],
}

SEARCH_QA_TASKS: dict[str, dict[str, Any]] = {
    "search_capital_france": {
        "prompt": "What is the capital of France?",
        "objective": "Single-hop NQ-style QA",
        "answers": {"paris"},
        "hints": ["search[france capital]", "search[france]", "answer[paris]"],
    },
    "search_python_creator": {
        "prompt": "Who created the Python programming language?",
        "objective": "Single-hop TriviaQA-style QA",
        "answers": {"guido van rossum", "guido"},
        "hints": ["search[python creator]", "search[guido van rossum]", "answer[guido van rossum]"],
    },
    "search_hotpot_bridge": {
        "prompt": "What river does Tower Bridge cross?",
        "objective": "Multi-hop HotpotQA-style QA",
        "answers": {"river thames", "thames"},
        "hints": ["search[tower bridge]", "search[bridge river thames]", "answer[river thames]"],
    },
}


@dataclass
class SearchQAEnv:
    """Text search-QA env with <search> / <answer> actions and rolling history."""

    task: TaskSpec
    state: str = ""
    step: int = 0
    done: bool = False
    success: bool = False
    _max_steps: int = 4
    _history: list[str] = field(default_factory=list)
    _accepted_answers: set[str] = field(default_factory=set)
    _hint_actions: list[str] = field(default_factory=list)
    _retriever: E5Retriever | None = None
    backend: str = "search_qa"

    @classmethod
    def create(cls, task_id: str = "search_capital_france") -> SearchQAEnv:
        spec = SEARCH_QA_TASKS.get(task_id, SEARCH_QA_TASKS["search_capital_france"])
        answers = {a.strip().lower() for a in spec["answers"]}
        return cls(
            task=TaskSpec(
                task_id=task_id,
                domain="search_qa",
                prompt=str(spec["prompt"]),
                objective=str(spec.get("objective", "")),
            ),
            _max_steps=4,
            _accepted_answers=answers,
            _hint_actions=list(spec.get("hints", [])),
            _retriever=E5Retriever.from_env(),
        )

    def reset(self) -> str:
        self.step = 0
        self.done = False
        self.success = False
        self._history = []
        self.state = "Search agent ready. No history yet."
        return self.state

    def memory_context(self) -> str:
        if not self._history:
            return "(empty)"
        return " ".join(self._history)

    def valid_actions(self) -> list[str]:
        if self.done:
            return []
        base = list(self._hint_actions)
        base.extend(["search[<query>]", "answer[<text>]"])
        return base

    def _lookup(self, query: str) -> str:
        retriever = self._retriever or E5Retriever.from_env()
        hits = retriever.search(query, SEARCH_QA_CORPUS, top_k=1)
        if hits:
            return f"Doc: {hits[0]}"
        return "Doc: No relevant information found."

    def step_action(self, action: str) -> tuple[str, float, bool]:
        if self.done:
            return self.state, 0.0, True
        self.step += 1
        act = action.strip()
        sm = _SEARCH_RE.match(act)
        am = _ANSWER_RE.match(act)
        reward = 0.0

        if sm:
            query = sm.group(1).strip()
            doc = self._lookup(query)
            self._history.append(f"<search>{query}</search><information>{doc}</information>")
            self.state = f"History: {self.memory_context()}"
        elif am:
            ans = am.group(1).strip().lower()
            if any(a in ans or ans in a for a in self._accepted_answers):
                self.done = True
                self.success = True
                reward = 1.0
                self.state = "Correct. Task success."
            else:
                self.done = True
                self.success = False
                self.state = "Incorrect answer."
        else:
            self.state = f"Invalid action format: {action}. Use search[...] or answer[...]."

        if self.step >= self._max_steps and not self.done:
            self.done = True
            self.state = "Episode ended: max steps."
        return self.state, reward, self.done


def list_search_qa_task_ids() -> list[str]:
    return sorted(SEARCH_QA_TASKS.keys())


def search_qa_task_prompts() -> dict[str, str]:
    return {tid: str(spec["prompt"]) for tid, spec in SEARCH_QA_TASKS.items()}
