"""Framework card, Table 2/3 anchors, CPU demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.xvec_emo.arithmetic import apply_tau, cosine, emotion_tau, interpolate, l2_norm
from ltx_trainer.xvec_emo.config import XvecEmoConfig
from ltx_trainer.xvec_emo.elimination import ELIMINATION_STEPS, localized_operand
from ltx_trainer.xvec_emo.metrics import delta_eecs, meets_identity_floor


def framework_card(cfg: XvecEmoConfig | None = None) -> dict[str, Any]:
    c = cfg or XvecEmoConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "backbone": c.backbone,
        "github": c.github,
        "localized_operand": localized_operand(),
    }


def table1_elimination() -> list[dict[str, Any]]:
    return [
        {
            "step": s.step,
            "operand": s.operand,
            "intervention": s.intervention,
            "outcome": s.outcome,
            "supports_arithmetic": s.supports_arithmetic,
        }
        for s in ELIMINATION_STEPS
    ]


def table2_en_held_out(cfg: XvecEmoConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or XvecEmoConfig()
    return [
        {
            "emotion": "angry",
            "base_eecs": c.base_eecs_angry,
            "avg4spk_eecs": c.avg4spk_eecs_angry,
            "delta_eecs": round(c.avg4spk_eecs_angry - c.base_eecs_angry, 3),
        },
        {"emotion": "happy", "avg4spk_eecs": c.avg4spk_eecs_happy},
        {"emotion": "sad", "avg4spk_eecs": c.avg4spk_eecs_sad},
    ]


def table3_ptbr(cfg: XvecEmoConfig | None = None) -> dict[str, Any]:
    c = cfg or XvecEmoConfig()
    return {
        "speakers": list(c.emouerj_speakers),
        "delta_eecs_mean": c.delta_eecs_ptbr,
        "secs_w_avg4spk_min": c.secs_w_avg4spk_min,
        "wer_mean": c.wer_ptbr_mean,
    }


def benchmarks_bundle(cfg: XvecEmoConfig | None = None) -> dict[str, Any]:
    c = cfg or XvecEmoConfig()
    return {
        "elimination": table1_elimination(),
        "en_held_out": table2_en_held_out(c),
        "ptbr": table3_ptbr(c),
        "geometry": {
            "x_neutral_norm": c.x_neutral_norm,
            "x_angry_norm": c.x_angry_norm,
            "cos_neutral_angry": c.cos_neutral_angry,
            "tau_over_x_ratio": c.tau_over_x_ratio,
        },
        "gains": {
            "delta_eecs_en_avg4spk": c.delta_eecs_en_avg4spk,
            "delta_eecs_en_single": c.delta_eecs_en_single,
            "delta_eecs_ptbr": c.delta_eecs_ptbr,
        },
    }


def _unit_vec(scale: float, dim: int, seed: int) -> list[float]:
    vals = [((seed + i) % 7 - 3) * 0.01 * scale for i in range(dim)]
    n = l2_norm(vals) or 1.0
    return [v / n * scale for v in vals]


def pipeline_demo(seed: int = 42, cfg: XvecEmoConfig | None = None) -> dict[str, Any]:
    c = cfg or XvecEmoConfig()
    dim = 8

    neutrals = [_unit_vec(c.x_neutral_norm, dim, seed + i) for i in range(4)]
    angries = [interpolate(n, apply_tau(n, _unit_vec(1.0, dim, seed), c.tau_over_x_ratio), 1.0) for n in neutrals]
    tau = emotion_tau(angries, neutrals)

    target = neutrals[0]
    x_new = apply_tau(target, tau, alpha=1.5)

    angry_gain = delta_eecs(c.base_eecs_angry, c.avg4spk_eecs_angry)
    avg4spk_beats_single = c.avg4spk_secs_mean > c.single_secs_mean

    return {
        "elimination_steps": len(ELIMINATION_STEPS),
        "localized_operand": localized_operand(),
        "tau_norm_ratio": round(l2_norm(tau) / (l2_norm(target) + 1e-9), 3),
        "cos_neutral_angry_demo": round(cosine(neutrals[0], angries[0]), 3),
        "x_new_differs_from_target": l2_norm(x_new) != l2_norm(target),
        "en_delta_eecs_avg4spk": round(angry_gain, 3),
        "en_delta_matches_paper": abs(angry_gain - 0.386) < 0.01,
        "avg4spk_identity_better": avg4spk_beats_single,
        "ptbr_delta_eecs": c.delta_eecs_ptbr,
        "secs_floor_met": meets_identity_floor(c.secs_w_avg4spk_min, floor=0.88),
        "formula": "x_new = x(target, neutral) + alpha * tau_emo",
    }


def evaluation_demo(seed: int = 42, cfg: XvecEmoConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
