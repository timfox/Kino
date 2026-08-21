"""Toy UNIVID caption + moderation smoke (arXiv:2606.05748)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.univid.capbench import sample_capbench_cases
from ltx_trainer.univid.metrics import capbench_bundle
from ltx_trainer.univid.moderation import (
    FusionNetwork,
    RiskFilter,
    TrendHead,
    UnividLite,
    UnividRAG,
    VideoSignals,
)
from ltx_trainer.univid.prompts import CAPTION_PROMPT


class ToyUnividCaptioner:
    """Policy-aware caption stub without VLM weights."""

    def caption(self, *, domain: str = "Regulated Act", ocr: str = "", seed: int = 0) -> str:
        rng = np.random.default_rng(seed)
        if "motorcycle" in ocr.lower() or domain == "Regulated Act":
            return (
                "The video shows three men riding together on a single red motorcycle along a "
                "paved road. None of the men wear helmets. Trees pass by in the background. "
                "The video may violate: High-Risk & Regulated Activities."
            )
        if domain == "Violence":
            return "The video depicts physical altercation between two adults in a public space."
        return f"A short-form clip with {domain.lower()} related visual content. OCR: {ocr or 'none'}."


def run_risk_filter_smoke(*, seed: int = 0) -> dict[str, Any]:
    cap = ToyUnividCaptioner()
    rf = RiskFilter(seed=seed)
    signals = VideoSignals(
        video_id="v-001",
        caption=cap.caption(domain="Regulated Act", ocr="night ride"),
        title="motorbike trio",
        ocr="no helmets",
    )
    result = rf.score(signals)
    return {
        "caption_prompt_len": len(CAPTION_PROMPT),
        "high_risk": result.high_risk,
        "routed": result.routed,
        "top_policy": max(result.policy_scores, key=result.policy_scores.get),
    }


def run_lite_rag_smoke(*, seed: int = 0) -> dict[str, Any]:
    cap = ToyUnividCaptioner()
    signals = VideoSignals(
        video_id="v-002",
        caption=cap.caption(domain="Regulated Act"),
        ocr="three on motorcycle without helmets",
        title="dangerous ride",
    )
    lite = UnividLite(seed=seed)
    rag = UnividRAG(lite=lite, seed=seed)
    lite_out = lite.decide(signals)
    rag_out = rag.decide(signals)
    return {
        "lite_decision": lite_out.decision.value,
        "lite_policy": lite_out.policy,
        "rag_decision": rag_out["decision"],
        "rag_retrieved_n": len(rag_out["retrieved"]),
        "rag_recoverable": rag_out["decision"] == "Violation",
    }


def run_capbench_smoke(*, seed: int = 0) -> dict[str, Any]:
    cap = ToyUnividCaptioner()
    cases = sample_capbench_cases(seed=seed)
    case = next(c for c in cases if c.domain == "Regulated Act")
    pred_caption = cap.caption(domain=case.domain)
    pred_events = [
        "The three men ride together on a single red motorcycle.",
        "None of the three men wear helmets while riding the motorcycle.",
    ]
    metrics = capbench_bundle(
        case.ground_truth_events,
        pred_events,
        violative_gt=case.violative_events,
        non_violative_gt=tuple(e for e in case.ground_truth_events if e not in case.violative_events),
    )
    return {"domain": case.domain, "metrics": metrics, "predicted_caption_words": len(pred_caption.split())}


def run_trend_smoke(*, seed: int = 0) -> dict[str, Any]:
    fusion = FusionNetwork(seed=seed)
    head = TrendHead(seed=seed)
    rng = np.random.default_rng(seed)
    n_pos, n_neg = 8, 40
    embs = []
    labels = []
    for i in range(n_pos + n_neg):
        sig = VideoSignals(
            video_id=f"t-{i}",
            caption="hot water challenge" if i < n_pos else "cooking pasta safely",
        )
        embs.append(fusion.encode(sig))
        labels.append(1.0 if i < n_pos else 0.0)
    fit = head.fit_few_shot(np.stack(embs), np.asarray(labels))
    test_emb = fusion.encode(VideoSignals(video_id="test", caption="hot water pouring challenge"))
    return {"few_shot_n": n_pos + n_neg, "fit": fit, "trend_score": round(head.predict(test_emb), 3)}


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    risk = run_risk_filter_smoke(seed=seed)
    actor = run_lite_rag_smoke(seed=seed + 1)
    bench = run_capbench_smoke(seed=seed + 2)
    trend = run_trend_smoke(seed=seed + 3)
    return {
        "risk_filter": risk,
        "moderation_actor": actor,
        "capbench": bench,
        "trend_governance": trend,
        "pipeline_ok": risk["high_risk"] and actor["rag_decision"] == "Violation",
    }
