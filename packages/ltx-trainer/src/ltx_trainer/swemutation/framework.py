"""Locate / Mutation / Judge / Self-Play modules (Sec. 3.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.swemutation.mutation import JudgeVerdict, judge_mutant, self_play_select


@dataclass
class LocateScope:
    """Files modifiable = golden solution files; F2P execution trace (Sec. 3.1.2)."""

    allowed_files: tuple[str, ...]
    f2p_trace_nodes: tuple[str, ...] = ()


@dataclass
class MutantCandidate:
    """One agentic mutant with metadata."""

    diff: str
    strategy_group: str
    explanation: str
    verdict: JudgeVerdict | None = None
    survival_count: int = 0


@dataclass
class AgenticMutationPipeline:
    """Four-module framework orchestration (smoke-level, no real repo I/O)."""

    locate: LocateScope
    candidates: list[MutantCandidate] = field(default_factory=list)

    def add_candidate(self, candidate: MutantCandidate) -> None:
        self.candidates.append(candidate)

    def run_judge(self, candidate: MutantCandidate, *, compiles: bool, fails_f2p: bool) -> bool:
        verdict = judge_mutant(
            modified_only_golden_files=True,
            compiles=compiles,
            fails_at_least_one_f2p=fails_f2p,
        )
        candidate.verdict = verdict
        return verdict.valid

    def run_self_play(self, *, threshold: int = 3) -> list[MutantCandidate]:
        counts = [c.survival_count for c in self.candidates]
        indices = self_play_select(counts, threshold=threshold)
        return [self.candidates[i] for i in indices]

    def summary(self) -> dict[str, Any]:
        valid = [c for c in self.candidates if c.verdict and c.verdict.valid]
        return {
            "allowed_files": list(self.locate.allowed_files),
            "candidates": len(self.candidates),
            "valid_mutants": len(valid),
            "strategies": list({c.strategy_group for c in self.candidates}),
        }
