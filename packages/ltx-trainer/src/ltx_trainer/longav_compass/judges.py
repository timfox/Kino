"""MLLM-assisted evaluation stubs (Gemini 3.1 Pro protocol, Sec. 4.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.longav_compass.annotation import BenchmarkCase
from ltx_trainer.longav_compass.metrics import (
    VQSubscores,
    clip_image_alignment,
    clip_text_video_alignment,
    score_event_vqa,
    transition_algorithm_score,
)


def _feature_from_seed(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim)


def judge_event_vqa(
    case: BenchmarkCase,
    *,
    model_name: str = "stub",
    quality_bias: float = 0.85,
) -> list[dict[str, Any]]:
    """Simulate per-event QA checklist judgments."""
    out: list[dict[str, Any]] = []
    for ev in case.events:
        answers = []
        for q in ev.qa_questions or [{"id": "default", "prompt": ev.action}]:
            # Deterministic pseudo-judge from case + event hash
            p = abs(hash((case.case_id, ev.event_id, q.get("id", "")))) % 100
            if p < int(100 * quality_bias):
                answers.append("yes")
            elif p < int(100 * quality_bias) + 8:
                answers.append("partial")
            else:
                answers.append("no")
        out.append(
            {
                "event_id": ev.event_id,
                "answers": answers,
                "vqa_score": score_event_vqa(answers),
                "judge": model_name,
            }
        )
    return out


def judge_segment_vq(
    case: BenchmarkCase,
    *,
    quality_bias: float = 0.8,
) -> list[dict[str, Any]]:
    """Four-axis VQ MOS stub per event."""
    rows = []
    for ev in case.events:
        base = 2.5 + 2.5 * quality_bias
        jitter = (abs(hash(ev.action)) % 7) * 0.05
        sub = VQSubscores(
            motion_naturalness=min(5.0, base + jitter),
            subject_integrity=min(5.0, base),
            artifact_control=min(5.0, base - 0.1),
            visual_fidelity=min(5.0, base + 0.05),
        )
        rows.append({"event_id": ev.event_id, "vq_mos": round(sub.mean(), 3), "subscores": sub.__dict__})
    return rows


def judge_full_video(
    case: BenchmarkCase,
    frame_features: list[np.ndarray],
    *,
    has_audio: bool = True,
    reference_image_feature: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full-video continuity, holism, alignment, and optional I2AV metrics."""
    text_feat = _feature_from_seed(abs(hash(case.global_description)) % (2**32))
    tv = clip_text_video_alignment(case.global_description, frame_features, text_feature=text_feat)
    trans = transition_algorithm_score(black_frames=False, flicker=False)
    cont = min(5.0, 2.5 + 2.0 * tv + (0.3 if case.complexity in ("L3", "L4") else 0.5))
    hol = min(5.0, cont - 0.15)
    out: dict[str, Any] = {
        "Cont.": round(cont, 3),
        "Trans.": round(trans, 3),
        "Hol.": round(hol, 3),
        "TVAlign": round(tv, 4),
    }
    if reference_image_feature is not None:
        out["IV1"] = round(0.95 + 0.04 * tv, 4)
        out["ImgAlign"] = round(clip_image_alignment(reference_image_feature, frame_features), 4)
    if has_audio:
        out["AVS"] = round(min(5.0, 2.8 + 1.5 * tv), 3)
        out["AudQ"] = round(min(5.0, 2.7 + 1.4 * tv), 3)
        out["AudL"] = round(min(5.0, 2.6 + 1.3 * tv), 3)
    return out


def evaluate_case(
    case: BenchmarkCase,
    *,
    n_frames: int = 16,
    has_audio: bool = True,
    quality_bias: float = 0.88,
) -> dict[str, Any]:
    """Run hierarchical evaluation on one benchmark case (stub)."""
    seed = abs(hash(case.case_id)) % (2**32)
    frame_feats = [_feature_from_seed(seed + i) for i in range(n_frames)]
    ref_img = _feature_from_seed(seed + 999) if case.task == "I2AV" else None

    vqa_rows = judge_event_vqa(case, quality_bias=quality_bias)
    vq_rows = judge_segment_vq(case, quality_bias=quality_bias)
    full = judge_full_video(case, frame_feats, has_audio=has_audio, reference_image_feature=ref_img)

    vqa_mean = float(np.mean([r["vqa_score"] for r in vqa_rows]))
    vq_mean = float(np.mean([r["vq_mos"] for r in vq_rows]))

    return {
        "case_id": case.case_id,
        "task": case.task,
        "event_vqa": vqa_rows,
        "event_vq": vq_rows,
        "aggregate": {
            "VQA": round(vqa_mean, 4),
            "VQ": round(vq_mean, 3),
            **full,
        },
    }


def evaluate_examples(
    cases: list[BenchmarkCase] | None = None,
    *,
    has_audio: bool = True,
    quality_bias: float = 0.88,
) -> list[dict[str, Any]]:
    """Evaluate all bundled example cases (T2AV, I2AV, V2AV)."""
    if cases is None:
        from ltx_trainer.longav_compass.annotation import (
            example_i2av_performance_ads_l4,
            example_t2av_performance_ads_l4,
            example_v2av_content_creator_l4,
        )

        cases = [
            example_t2av_performance_ads_l4(),
            example_i2av_performance_ads_l4(),
            example_v2av_content_creator_l4(),
        ]
    return [evaluate_case(c, has_audio=has_audio, quality_bias=quality_bias) for c in cases]
