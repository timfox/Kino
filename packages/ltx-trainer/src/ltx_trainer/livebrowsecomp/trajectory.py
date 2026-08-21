"""Search trajectory analysis (§2.3 model- vs retrieval-originated queries)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

Origin = Literal["model", "retrieval", "unknown"]

_TOKEN = re.compile(r"[a-z0-9]{4,}", re.I)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text)}


def _query_entities(query: str) -> set[str]:
    """Heuristic salient tokens from a search query."""
    stop = {
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "the",
        "and",
        "for",
        "from",
        "with",
        "about",
        "search",
        "find",
    }
    return {t for t in _tokens(query) if t not in stop and len(t) >= 4}


@dataclass
class RetrievalHit:
    doc_id: str = ""
    snippet: str = ""
    is_gold: bool = False
    is_evidence: bool = False


@dataclass
class SearchRound:
    round_idx: int
    query: str
    reasoning_before: str = ""
    hits: list[RetrievalHit] = field(default_factory=list)
    reasoning_after: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_idx,
            "query": self.query,
            "hits": [h.__dict__ for h in self.hits],
        }


@dataclass
class SearchTrajectory:
    question_id: str
    rounds: list[SearchRound] = field(default_factory=list)
    final_answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_answer": self.final_answer,
        }


def query_origin(
    query: str,
    *,
    prior_reasoning: str,
    prior_retrieval_text: str,
) -> Origin:
    """
    Where key query terms first appear (§2.3).

    If salient query tokens appear in prior retrieval but not prior reasoning → retrieval.
    If they appear in reasoning first → model.
    """
    ents = _query_entities(query)
    if not ents:
        return "unknown"
    r_tok = _tokens(prior_retrieval_text)
    m_tok = _tokens(prior_reasoning)
    in_r = sum(1 for e in ents if e in r_tok)
    in_m = sum(1 for e in ents if e in m_tok)
    if in_r > in_m:
        return "retrieval"
    if in_m > 0:
        return "model"
    if in_r > 0:
        return "retrieval"
    return "model"


def evidence_used_after_retrieval(
    round_: SearchRound,
    *,
    lookahead_reasoning: str,
    window_chars: int = 4000,
) -> bool:
    """Gold/evidence hit counts as used if terms appear in subsequent reasoning/answer."""
    gold_snips = [h.snippet for h in round_.hits if h.is_gold or h.is_evidence]
    if not gold_snips:
        return False
    pool = (round_.reasoning_after + "\n" + lookahead_reasoning)[:window_chars].lower()
    for snip in gold_snips:
        for tok in _tokens(snip):
            if len(tok) >= 5 and tok in pool:
                return True
    return False


@dataclass
class TrajectoryReport:
    num_queries: int
    model_originated: int
    retrieval_originated: int
    model_originated_rate: float
    gold_retrievals: int
    gold_used: int
    evidence_use_rate: float
    by_progress: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "model_originated": self.model_originated,
            "retrieval_originated": self.retrieval_originated,
            "model_originated_rate": round(self.model_originated_rate, 4),
            "gold_retrievals": self.gold_retrievals,
            "gold_used": self.gold_used,
            "evidence_use_rate": round(self.evidence_use_rate, 4),
            "by_progress": self.by_progress,
        }


def analyze_trajectory(traj: SearchTrajectory) -> TrajectoryReport:
    prior_reasoning = ""
    prior_retrieval = ""
    model_n = ret_n = 0
    gold_ret = gold_used = 0
    progress_buckets: list[dict[str, Any]] = []

    for i, rnd in enumerate(traj.rounds):
        origin = query_origin(
            rnd.query,
            prior_reasoning=prior_reasoning,
            prior_retrieval_text=prior_retrieval,
        )
        if origin == "model":
            model_n += 1
        elif origin == "retrieval":
            ret_n += 1

        lookahead = ""
        if i + 1 < len(traj.rounds):
            lookahead = traj.rounds[i + 1].reasoning_before
        elif traj.final_answer:
            lookahead = traj.final_answer

        if any(h.is_gold or h.is_evidence for h in rnd.hits):
            gold_ret += 1
            if evidence_used_after_retrieval(rnd, lookahead_reasoning=lookahead):
                gold_used += 1

        prior_reasoning += "\n" + rnd.reasoning_before + "\n" + rnd.reasoning_after
        prior_retrieval += "\n" + "\n".join(h.snippet for h in rnd.hits)

        total = model_n + ret_n
        if total:
            progress_buckets.append(
                {
                    "round": rnd.round_idx,
                    "progress_pct": round(100 * (i + 1) / max(1, len(traj.rounds)), 1),
                    "model_originated_rate": round(model_n / total, 4),
                }
            )

    nq = model_n + ret_n
    return TrajectoryReport(
        num_queries=nq,
        model_originated=model_n,
        retrieval_originated=ret_n,
        model_originated_rate=model_n / nq if nq else 0.0,
        gold_retrievals=gold_ret,
        gold_used=gold_used,
        evidence_use_rate=gold_used / gold_ret if gold_ret else 0.0,
        by_progress=progress_buckets,
    )


def analyze_trajectories(trajectories: list[SearchTrajectory]) -> dict[str, Any]:
    reports = [analyze_trajectory(t) for t in trajectories]
    nq = sum(r.num_queries for r in reports)
    mo = sum(r.model_originated for r in reports)
    gr = sum(r.gold_retrievals for r in reports)
    gu = sum(r.gold_used for r in reports)
    return {
        "trajectories": len(trajectories),
        "aggregate": {
            "num_queries": nq,
            "model_originated_rate": mo / nq if nq else 0.0,
            "evidence_use_rate": gu / gr if gr else 0.0,
        },
        "per_trajectory": [r.to_dict() for r in reports],
    }
