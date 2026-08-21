"""Critic agent — evidence sufficiency + cross-principle consistency (§4.4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.agentfair.agents import EvaluationResult, evaluate_subprinciple
from ltx_trainer.agentfair.config import AgentFAIRConfig
from ltx_trainer.agentfair.extract import MetadataRecord


@dataclass
class CriticDecision:
    principle: str
    retry: bool
    reasons: list[str]
    revised: EvaluationResult | None = None


def hard_check_fails(principle: str, result: EvaluationResult, meta: MetadataRecord) -> list[str]:
    """Deterministic logical checks used to trigger retry."""
    fails: list[str] = []
    score = result.score
    if score < 0 or score > 3:
        fails.append("score_out_of_range")
    if score > 0 and not result.evidence and not meta.snippets:
        fails.append("nonzero_without_evidence")
    d = meta.deterministic
    if principle == "F1" and score >= 2 and not d.get("identifier_present"):
        fails.append("f1_score_without_pid")
    if principle == "F1" and score >= 3 and not d.get("https_landing"):
        fails.append("f1_full_without_https")
    if principle == "R1.1" and score >= 2 and not d.get("license_present"):
        fails.append("r11_without_license")
    if principle == "A1.1" and score >= 2 and not (
        d.get("https_landing") or meta.fields.get("protocol")
    ):
        fails.append("a11_without_protocol")
    return fails


def consistency_flags(results: dict[str, EvaluationResult]) -> list[str]:
    """Cross-sub-principle contradictions that warrant review."""
    flags: list[str] = []
    a11 = results.get("A1.1")
    a12 = results.get("A1.2")
    if a11 and a12 and a11.score >= 3 and a12.score <= 1:
        flags.append("open_protocol_vs_unclear_auth")
    f1 = results.get("F1")
    f3 = results.get("F3")
    if f1 and f3 and f1.score >= 3 and f3.score == 0:
        flags.append("strong_pid_without_distribution")
    i1 = results.get("I1")
    i2 = results.get("I2")
    if i1 and i2 and i1.score >= 2 and i2.score == 0:
        flags.append("kr_without_vocabulary_grounding")
    return flags


def critic_review(
    principle: str,
    result: EvaluationResult,
    meta: MetadataRecord,
    *,
    config: AgentFAIRConfig | None = None,
    all_results: dict[str, EvaluationResult] | None = None,
) -> CriticDecision:
    cfg = config or AgentFAIRConfig()
    reasons = hard_check_fails(principle, result, meta)
    thr = cfg.i1_confidence_threshold if principle == "I1" else cfg.confidence_threshold
    if result.confidence < thr:
        reasons.append(f"low_confidence<{thr}")
    if all_results:
        for flag in consistency_flags(all_results):
            if principle in flag or principle.split(".")[0] in flag.upper():
                reasons.append(flag)
            elif flag.startswith(principle.lower().replace(".", "")):
                reasons.append(flag)
            elif principle in {"A1.1", "A1.2"} and "auth" in flag:
                reasons.append(flag)
            elif principle in {"F1", "F3"} and "pid" in flag:
                reasons.append(flag)
            elif principle in {"I1", "I2"} and "vocabulary" in flag:
                reasons.append(flag)

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    if not uniq or not cfg.enable_critic:
        return CriticDecision(principle=principle, retry=False, reasons=uniq)

    # Targeted re-evaluation with the same deterministic agent (paper: critic repair)
    revised = evaluate_subprinciple(principle, meta)
    # Conservative nudge: if hard check failed for missing evidence, clamp score
    if "nonzero_without_evidence" in uniq and revised.score > 0 and not revised.evidence:
        revised.score = 0
        revised.recommendations = list(revised.recommendations) + [
            "Critic: non-zero score rejected without citeable evidence."
        ]
        revised.confidence = min(revised.confidence, 0.4)
    if "f1_score_without_pid" in uniq:
        revised.score = min(revised.score, 1)
        revised.recommendations = list(revised.recommendations) + [
            "Critic: lower F1 until a PID is observed."
        ]
    return CriticDecision(principle=principle, retry=True, reasons=uniq, revised=revised)
