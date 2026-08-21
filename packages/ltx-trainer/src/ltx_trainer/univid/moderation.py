"""Three-stage UNIVID moderation pipeline stubs (arXiv:2606.05748)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ltx_trainer.univid.capbench import VKB_EXAMPLES
from ltx_trainer.univid.prompts import format_lite_prompt, format_rag_prompt


class ModerationDecision(str, Enum):
    APPROVE = "Approve"
    VIOLATION = "Violation"


@dataclass
class VideoSignals:
    video_id: str
    caption: str
    title: str = ""
    ocr: str = ""
    asr: str = ""
    region: str = "global"
    nickname: str = ""
    profile: str = ""
    fusion_embedding: np.ndarray | None = None


@dataclass
class PolicyHeadConfig:
    name: str
    threshold: float = 0.65


@dataclass
class RiskFilterResult:
    video_id: str
    policy_scores: dict[str, float]
    high_risk: bool
    routed: bool


class FusionNetwork:
    """Toy fusion of caption + OCR/title/risk signals into shared embedding."""

    def __init__(self, *, dim: int = 32, seed: int = 0) -> None:
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._proj = self._rng.normal(0, 0.1, size=(dim, dim))

    def encode(self, signals: VideoSignals) -> np.ndarray:
        if signals.fusion_embedding is not None:
            return np.asarray(signals.fusion_embedding, dtype=np.float64)
        text = " ".join([signals.caption, signals.title, signals.ocr, signals.asr]).lower()
        vec = np.zeros(self.dim)
        for i, word in enumerate(text.split()[: self.dim]):
            vec[i % self.dim] += hash(word) % 997 / 997.0
        return self._proj @ vec


class RiskFilter:
    """Stage A — multi-modal risk funnel with policy MLP heads."""

    def __init__(
        self,
        policy_heads: list[PolicyHeadConfig] | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.fusion = FusionNetwork(seed=seed)
        heads = policy_heads or [
            PolicyHeadConfig("Violence", 0.55),
            PolicyHeadConfig("Sexual Abuse", 0.60),
            PolicyHeadConfig("Integrity", 0.58),
            PolicyHeadConfig("Dangerous Driving", 0.52),
        ]
        self.heads = {h.name: h for h in heads}
        rng = np.random.default_rng(seed)
        self._weights = {name: rng.normal(0, 0.05, size=32) for name in self.heads}

    def score(self, signals: VideoSignals) -> RiskFilterResult:
        emb = self.fusion.encode(signals)
        text = signals.caption.lower()
        risk_cues = ("violence", "nudity", "motorcycle", "helmet", "fraud", "weapon", "blood")
        cue_boost = 0.15 * sum(1 for c in risk_cues if c in text)
        scores = {}
        for name, head in self.heads.items():
            raw = float(np.tanh(np.dot(self._weights[name], emb)) * 0.5 + 0.5 + cue_boost)
            scores[name] = round(min(raw, 0.99), 3)
        high_risk = any(scores[n] >= self.heads[n].threshold for n in scores)
        return RiskFilterResult(
            video_id=signals.video_id,
            policy_scores=scores,
            high_risk=high_risk,
            routed=high_risk,
        )


@dataclass
class LiteDecision:
    decision: ModerationDecision
    policy: str | None
    rationale: str


class UnividLite:
    """Stage B — unified moderation actor (autoregressive Approve/Violation + policy)."""

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def decide(self, signals: VideoSignals) -> LiteDecision:
        text = " ".join([signals.caption, signals.ocr, signals.asr, signals.title]).lower()
        violative_cues = {
            "Dangerous Driving": ("motorcycle", "helmet", "without helmet", "speedometer"),
            "Nudity & Sexual Activity": ("nudity", "sexual", "suggestive"),
            "Harassment & Hateful Behavior": ("email", "verification code", "dox"),
            "High-Risk & Regulated Activities": ("riding on", "vehicle exterior", "convoy"),
        }
        for policy, cues in violative_cues.items():
            if any(c in text for c in cues):
                return LiteDecision(
                    decision=ModerationDecision.VIOLATION,
                    policy=policy,
                    rationale=f"Detected cues for {policy} in multimodal context.",
                )
        if self._rng.random() < 0.08:
            return LiteDecision(
                decision=ModerationDecision.VIOLATION,
                policy="Integrity",
                rationale="Latent risk signal from contextual inference.",
            )
        return LiteDecision(
            decision=ModerationDecision.APPROVE,
            policy=None,
            rationale="No policy violation detected.",
        )

    def prompt(self, signals: VideoSignals) -> str:
        return format_lite_prompt(
            region=signals.region,
            title=signals.title,
            nickname=signals.nickname,
            asr=signals.asr,
            ocr=signals.ocr,
            profile=signals.profile,
        )


class ViolationKnowledgeBase:
    """Structured violative events for UNIVID-RAG retrieval."""

    def __init__(self, events: list[dict[str, str]] | None = None) -> None:
        self.events = events or list(VKB_EXAMPLES)
        self._embeddings: list[np.ndarray] = []

    def build(self, encoder: FusionNetwork) -> None:
        self._embeddings = []
        for ev in self.events:
            sig = VideoSignals(video_id="vkb", caption=ev["event"])
            self._embeddings.append(encoder.encode(sig))

    def retrieve(self, query_emb: np.ndarray, *, top_k: int = 3) -> list[tuple[str, str, float]]:
        if not self._embeddings:
            return []
        sims = []
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        for i, emb in enumerate(self._embeddings):
            e = emb / (np.linalg.norm(emb) + 1e-9)
            sims.append((float(np.dot(q, e)), i))
        sims.sort(reverse=True)
        out: list[tuple[str, str, float]] = []
        for sim, idx in sims[:top_k]:
            ev = self.events[idx]
            out.append((ev["policy"], ev["event"], round(sim, 3)))
        return out


class UnividRAG:
    """Stage B leakage mitigation — top-k VKB retrieval + in-context moderation."""

    def __init__(self, lite: UnividLite | None = None, *, seed: int = 0) -> None:
        self.lite = lite or UnividLite(seed=seed)
        self.encoder = FusionNetwork(seed=seed + 1)
        self.vkb = ViolationKnowledgeBase()
        self.vkb.build(self.encoder)

    def decide(self, signals: VideoSignals, *, top_k: int = 3) -> dict[str, Any]:
        emb = self.encoder.encode(signals)
        retrieved = self.vkb.retrieve(emb, top_k=top_k)
        lite_dec = self.lite.decide(signals)
        # RAG increases sensitivity on low-similarity hard cases
        rag_boost = any(sim > 0.2 for _, _, sim in retrieved)
        decision = lite_dec.decision
        policy = lite_dec.policy
        if rag_boost and decision == ModerationDecision.APPROVE and retrieved:
            top_policy = retrieved[0][0]
            decision = ModerationDecision.VIOLATION
            policy = top_policy
        prompt = format_rag_prompt([(p, r) for p, r, _ in retrieved], region=signals.region)
        return {
            "decision": decision.value,
            "policy": policy,
            "retrieved": [{"policy": p, "reason": r, "similarity": s} for p, r, s in retrieved],
            "prompt_chars": len(prompt),
            "lite_would_approve": lite_dec.decision == ModerationDecision.APPROVE,
        }


class TrendHead:
    """Stage C — few-shot MLP on cached fusion embeddings."""

    def __init__(self, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.1, size=32)
        self.bias = -0.1

    def fit_few_shot(self, embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        x = np.asarray(embeddings, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.shape[0] < 2:
            return {"train_acc": 0.0}
        design = np.hstack([x, np.ones((x.shape[0], 1))])
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        self.w = coef[:-1]
        self.bias = float(coef[-1])
        preds = (x @ self.w + self.bias) > 0.5
        acc = float(np.mean(preds == (y > 0.5)))
        return {"train_acc": round(acc, 3)}

    def predict(self, embedding: np.ndarray) -> float:
        return float(1 / (1 + np.exp(-(float(np.dot(self.w, embedding)) + self.bias))))
