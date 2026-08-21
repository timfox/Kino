"""Witness-adjudicator sentence reward and GRPO helpers."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from ltx_trainer.vcap.config import VCapConfig


@dataclass
class JudgeScores:
    scorr: int
    scomp: int
    stxt: int
    analysis: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> JudgeScores:
        return cls(
            scorr=int(data.get("Correctness", data.get("scorr", 5))),
            scomp=int(data.get("Completeness", data.get("scomp", 5))),
            stxt=int(data.get("Text Quality", data.get("stxt", 5))),
            analysis=str(data.get("Analysis", "")),
        )


def parse_judge_json(text: str) -> JudgeScores:
    """Parse reward-model JSON output; fall back to regex extraction."""
    text = (text or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return JudgeScores.from_dict(data)
    except json.JSONDecodeError:
        pass
    nums = {
        "Correctness": _extract_score(text, "Correctness"),
        "Completeness": _extract_score(text, "Completeness"),
        "Text Quality": _extract_score(text, "Text Quality"),
    }
    return JudgeScores.from_dict(nums)


def _extract_score(text: str, key: str) -> int:
    m = re.search(rf'"{re.escape(key)}"\s*:\s*(\d+)', text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(rf"{re.escape(key)}\s*[:=]\s*(\d+)", text, re.I)
    return int(m.group(1)) if m else 5


def sentence_reward(scores: JudgeScores, cfg: VCapConfig | None = None) -> float:
    """Eq. (4): r = wcorr*scorr + wcomp*scomp + wtxt*stxt."""
    cfg = cfg or VCapConfig()
    return (
        cfg.wcorr * scores.scorr
        + cfg.wcomp * scores.scomp
        + cfg.wtxt * scores.stxt
    )


def video_reward(
    global_scores: JudgeScores,
    local_scores: JudgeScores | None,
    cfg: VCapConfig | None = None,
) -> float:
    """Eq. (5): r_video = r_global + w_local * r_local."""
    cfg = cfg or VCapConfig()
    r = sentence_reward(global_scores, cfg)
    if local_scores is not None:
        r += cfg.wlocal_video * sentence_reward(local_scores, cfg)
    return r


def mock_judge_scores(
    policy_caption: str,
    reference_caption: str,
    *,
    cfg: VCapConfig | None = None,
) -> JudgeScores:
    """Heuristic witness-adjudicator proxy when no live MLLM judge is available."""
    cfg = cfg or VCapConfig()
    ref_words = set(_content_words(reference_caption))
    pol_words = set(_content_words(policy_caption))
    if not ref_words:
        ref_words = pol_words
    overlap = len(ref_words & pol_words) / max(1, len(ref_words))
    extra = len(pol_words - ref_words) / max(1, len(pol_words))
    hallucination_penalty = min(1.0, extra * 1.2)
    scorr = int(round(cfg.score_max * (1.0 - 0.7 * hallucination_penalty)))
    scomp = int(round(cfg.score_max * overlap))
    stxt = 8 if len(policy_caption.split()) > 20 else 7
    return JudgeScores(
        scorr=max(cfg.score_min, min(cfg.score_max, scorr)),
        scomp=max(cfg.score_min, min(cfg.score_max, scomp)),
        stxt=max(cfg.score_min, min(cfg.score_max, stxt)),
        analysis=f"mock overlap={overlap:.2f} extra_ratio={extra:.2f}",
    )


def _content_words(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "with",
        "is", "are", "was", "were", "it", "this", "that", "for", "as", "by",
    }
    return {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text) if w.lower() not in stop}


def grpo_advantages(rewards: list[float], epsilon: float = 1e-6) -> list[float]:
    """Group-relative advantages (Eq. 6 style)."""
    if not rewards:
        return []
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = math.sqrt(var) + epsilon
    return [(r - mu) / sigma for r in rewards]


def estimate_fact_counts_from_captions(
    reference: str,
    policy: str,
    *,
    N: int = 100,
) -> tuple[int, int, int]:
    """Estimate (c, n_total, m) for hypergeometric demo from caption overlap."""
    ref = _content_words(reference)
    pol = _content_words(policy)
    m = max(1, len(ref))
    c = len(ref & pol)
    n_total = max(c + 1, len(pol))
    n_total = min(N, max(n_total, c + max(1, len(pol - ref))))
    return c, n_total, m
