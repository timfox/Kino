"""CS-NMG near-miss generation and tri-level filtering gate (§3)."""

from __future__ import annotations

from typing import Any, Literal

from ltx_trainer.cs_nmg.config import CsNmgConfig
from ltx_trainer.cs_nmg.poi import normalized_levenshtein

FilterMode = Literal["none", "acoustic", "ac_text", "ac_ph", "ac_ph_text"]


def acoustic_gate(
    score: float,
    nbest_scores: list[float],
    *,
    delta: float,
) -> bool:
    """log p(y|x) >= max N-best − Δ (Eq. 3)."""
    if not nbest_scores:
        return True
    return score >= max(nbest_scores) - delta


def text_phoneme_gate(
    candidate: str,
    reference_span: str,
    *,
    tau_text: float,
    tau_phoneme: float,
    phoneme_fn: Any = None,
) -> bool:
    """Hardness + plausibility: d_txt ≥ τ_txt and d_ph ≤ τ_ph (Eq. 6)."""
    d_txt = normalized_levenshtein(candidate, reference_span)
    phi = phoneme_fn or (lambda s: s.lower())
    d_ph = normalized_levenshtein(phi(candidate), phi(reference_span))
    return d_txt >= tau_text and d_ph <= tau_ph


def replace_poi_span(reference: str, poi_start: int, poi_end: int, replacement: str) -> str:
    tokens = reference.split()
    repl_tokens = replacement.split()
    out = tokens[:poi_start] + repl_tokens + tokens[poi_end + 1 :]
    return " ".join(out)


def filter_near_miss(
    *,
    score: float,
    nbest_scores: list[float],
    candidate_span: str,
    reference_span: str,
    mode: FilterMode = "ac_ph_text",
    delta: float = 4.0,
    tau_text: float = 0.4,
    tau_phoneme: float = 0.6,
) -> bool:
    if mode == "none":
        return True
    if not acoustic_gate(score, nbest_scores, delta=delta):
        return False
    if mode == "acoustic":
        return True
    if mode in {"ac_text", "ac_ph_text"}:
        if normalized_levenshtein(candidate_span, reference_span) < tau_text:
            return False
    if mode in {"ac_ph", "ac_ph_text"}:
        if normalized_levenshtein(candidate_span.lower(), reference_span.lower()) > tau_phoneme:
            return False
    return True


def near_miss_demo(*, cfg: CsNmgConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CsNmgConfig()
    ref = "enzyme 5 alpha reductase được tạo ra"
    ref_span = "reductase"
    candidates = [
        ("reduc tây", 0.5),
        ("ri đắc tê", -1.0),
        ("reductase", 2.0),
    ]
    nbest = [0.0, -0.5, -1.0]
    kept = [
        filter_near_miss(
            score=s,
            nbest_scores=nbest,
            candidate_span=c,
            reference_span=ref_span,
            mode="ac_ph_text",
            delta=cfg.acoustic_margin_delta,
            tau_text=cfg.tau_text,
            tau_phoneme=cfg.tau_phoneme,
        )
        for c, s in candidates
    ]
    return {
        "reference": ref,
        "candidates_tested": len(candidates),
        "tri_level_kept": sum(kept),
        "rejects_exact_match": not kept[-1],
        "keeps_hard_substitution": kept[0] or kept[1],
        "delta": cfg.acoustic_margin_delta,
    }
