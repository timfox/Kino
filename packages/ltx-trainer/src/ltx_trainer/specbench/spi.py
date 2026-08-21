"""Subject–Predicate–Impact decomposition for SpecBench matching (paper §2.3.3)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_STOP = frozenset(
    "a an the and or of in on for to with is are was were be been being "
    "that this these those it its their from as at by not no".split()
)

_SYNONYMS: dict[str, frozenset[str]] = {
    "workload": frozenset({"workload", "workloads", "gang", "podgroup", "pod group"}),
    "timeout": frozenset({"timeout", "schedulingtimeoutseconds", "timer", "expiry"}),
    "preemption": frozenset({"preemption", "preempt", "priority", "gang-level"}),
    "namespace": frozenset({"namespace", "namespaced", "cross-namespace", "tenancy"}),
    "status": frozenset({"status", "observability", "conditions", "events", "workloadstatus"}),
    "deadlock": frozenset({"deadlock", "retry", "requeue", "lifecycle", "competing"}),
    "membership": frozenset({"membership", "association", "podgroup", "selector", "pod-to"}),
    "autoscaler": frozenset({"autoscaler", "cluster autoscaler", "ca", "provisioner", "scale-up"}),
    "mutability": frozenset({"mutability", "immutable", "mutable", "update", "race"}),
    "admission": frozenset({"admission", "lifecycle", "reservation", "rollback", "permit"}),
}


@dataclass(frozen=True)
class SPITriple:
    subject: str
    predicate: str
    impact: str = ""

    def normalized_subject_tokens(self) -> set[str]:
        return _expand_synonyms(_tokenize(self.subject))

    def normalized_predicate_tokens(self) -> set[str]:
        return _expand_synonyms(_tokenize(self.predicate))


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def _expand_synonyms(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for tok in tokens:
        for group in _SYNONYMS.values():
            if tok in group:
                out |= group
    return out


def token_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def spi_subject_predicate_match(
    pred: SPITriple,
    gold: SPITriple,
    *,
    subject_threshold: float,
    predicate_threshold: float,
) -> bool:
    sub = token_overlap(pred.normalized_subject_tokens(), gold.normalized_subject_tokens())
    pred_tok = token_overlap(pred.normalized_predicate_tokens(), gold.normalized_predicate_tokens())
    return sub >= subject_threshold and pred_tok >= predicate_threshold


def decompose_deficiency(text: str) -> SPITriple:
    """Heuristic SPI decomposition for stub matching (no LLM)."""
    lower = text.lower()
    subject = "system specification"
    for key, syns in _SYNONYMS.items():
        if any(s in lower for s in syns):
            subject = key.replace("_", " ")
            break
    if "workload" in lower and "pod" in lower:
        subject = "pod workload reference"
    if "timeout" in lower or "schedulingtimeout" in lower:
        subject = "gang scheduling timeout"
    if "preemption" in lower:
        subject = "gang preemption semantics"
    if "namespace" in lower or "cross-namespace" in lower:
        subject = "workload namespace scoping"
    if "status" in lower or "observability" in lower or "workloadstatus" in lower:
        subject = "workload status observability"
    if "deadlock" in lower or "retry lifecycle" in lower:
        subject = "deadlock avoidance retry lifecycle"
    if "membership" in lower or "association" in lower or "podgroup" in lower:
        subject = "gang membership association"
    if "autoscaler" in lower or "cluster autoscaler" in lower:
        subject = "cluster autoscaler interaction"
    if "mutability" in lower or "immutable" in lower:
        subject = "workload spec mutability"
    if "admission" in lower:
        subject = "gang admission lifecycle"

    predicate = "underspecified"
    if "ambiguous" in lower or "unclear" in lower:
        predicate = "ambiguous semantics"
    elif "missing" in lower or "not described" in lower or "tbd" in lower:
        predicate = "missing specification"
    elif "inconsistent" in lower or "contradict" in lower:
        predicate = "internal inconsistency"
    elif "unsafe" in lower or "deadlock" in lower:
        predicate = "unsafe or incomplete lifecycle"
    elif "undefined" in lower:
        predicate = "undefined behavior"

    impact = ""
    if "deadlock" in lower:
        impact = "scheduling deadlocks"
    elif "debug" in lower or "observ" in lower:
        impact = "operator cannot diagnose failures"
    elif "preemption" in lower:
        impact = "premature workload disruption"
    elif "timeout" in lower:
        impact = "unpredictable resource release"
    elif "security" in lower or "tenant" in lower or "rbac" in lower:
        impact = "cross-tenant interference risk"

    return SPITriple(subject=subject, predicate=predicate, impact=impact)
