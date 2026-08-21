"""Contextual feature encoding — Table 2 (arXiv:2605.28707)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ltx_trainer.ethical_pluralism.simplex import NormativeScores, plurality_features, project_simplex
from ltx_trainer.ethical_pluralism.taxonomy import SUBTHEORY_BY_ID, SUBTHEORY_IDS, NormativeSchool

# Categorical vocabularies for one-hot encoding (stub benchmark)
SEVERITY_LEVELS = ("none", "low", "medium", "high")
DURATION_LEVELS = ("immediate", "short", "long", "permanent")
UTILITY_LEVELS = ("good", "bad", "gray")
INTENTION_LEVELS = ("benevolent", "neutral", "malicious", "mixed")
RELATIONSHIP_LEVELS = ("stranger", "friend", "family", "colleague", "authority", "dependent")
DOMAIN_LEVELS = ("work", "healthcare", "family", "public", "online", "legal", "education")


def _one_hot(value: str, vocabulary: tuple[str, ...]) -> list[float]:
    v = [0.0] * len(vocabulary)
    if value in vocabulary:
        v[vocabulary.index(value)] = 1.0
    return v


@dataclass
class EthicalCase:
    """Single Ethics Pluralism benchmark row (Table 2)."""

    case_id: str
    subtheory_id: str
    selftext: str
    summary: str = ""
    active_agent: str = ""
    passive_agent: str = ""
    relationship: str = "stranger"
    action: str = ""
    domain: str = "public"
    ethical_issues: str = ""
    consequence: str = ""
    severity: str = "medium"
    duration: str = "short"
    utility: str = "gray"
    moral_intention: str = "neutral"
    principles_upheld: str = ""
    principles_violated: str = ""
    moral_decision: str = "gray"
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    @property
    def school(self) -> NormativeSchool:
        return SUBTHEORY_BY_ID[self.subtheory_id].school

    @property
    def normative_scores(self) -> NormativeScores:
        return project_simplex(self.alpha, self.beta, self.gamma)

    def set_normative_from_school(self, *, noise: float = 0.0) -> None:
        """Assign simplex scores peaked on the case's normative school."""
        base: dict[NormativeSchool, NormativeScores] = {
            "consequentialism": (0.62, 0.18, 0.20),
            "virtue_ethics": (0.18, 0.62, 0.20),
            "deontology": (0.20, 0.18, 0.62),
        }
        a, b, g = base[self.school]
        a, b, g = a + noise, b + noise * 0.5, g + noise * 0.3
        self.alpha, self.beta, self.gamma = project_simplex(a, b, g)


def infer_normative_scores(case: EthicalCase) -> NormativeScores:
    """Use LLM priors when present; else school-aligned heuristic."""
    if case.alpha + case.beta + case.gamma > 0:
        return project_simplex(case.alpha, case.beta, case.gamma)
    case.set_normative_from_school()
    return case.normative_scores


def subtheory_lexicon_flags(case: EthicalCase) -> np.ndarray:
    """Lexical presence of bracketed subtheory markers in narrative (inference-time safe)."""
    blob = f"{case.selftext} {case.summary}"
    return np.array(
        [1.0 if f"[{sid}]" in blob else 0.0 for sid in SUBTHEORY_IDS],
        dtype=np.float64,
    )


def context_one_hot(case: EthicalCase) -> np.ndarray:
    """Eq. (5): C_sev, C_dur, C_util, MI, P_up, P_vi."""
    parts: list[float] = []
    parts.extend(_one_hot(case.severity, SEVERITY_LEVELS))
    parts.extend(_one_hot(case.duration, DURATION_LEVELS))
    parts.extend(_one_hot(case.utility, UTILITY_LEVELS))
    parts.extend(_one_hot(case.moral_intention, INTENTION_LEVELS))
    parts.extend(_one_hot(case.relationship, RELATIONSHIP_LEVELS))
    parts.extend(_one_hot(case.domain, DOMAIN_LEVELS))
    # principles as weak bag-of-flags
    upheld = 1.0 if case.principles_upheld.strip() else 0.0
    violated = 1.0 if case.principles_violated.strip() else 0.0
    parts.extend([upheld, violated])
    return np.array(parts, dtype=np.float64)


def case_feature_vector(
    case: EthicalCase,
    *,
    use_normative: bool = True,
    use_context: bool = True,
    use_embeddings: bool = True,
    include_lexicon: bool = True,
    projection_dim: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Unified feature space for stacked ensemble (normative + semantic streams)."""
    from ltx_trainer.ethical_pluralism.embeddings import project_supervector, triple_bert_supervector

    chunks: list[np.ndarray] = []
    scores = infer_normative_scores(case)
    if use_normative:
        pf = plurality_features(scores)
        chunks.append(np.array([scores[0], scores[1], scores[2], pf["margin"], pf["entropy"]]))
    if use_context:
        chunks.append(context_one_hot(case))
        if include_lexicon:
            chunks.append(subtheory_lexicon_flags(case))
    if use_embeddings:
        sv = triple_bert_supervector(case.selftext, case.summary)
        chunks.append(project_supervector(sv, projection_dim, seed=seed))
    if not chunks:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(chunks)
