"""MTLLFM framework card, paper tables, and smoke demos (arXiv:2605.25409)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.mtllfm.config import MTLLFMConfig
from ltx_trainer.mtllfm.gating import adaptive_modality_gating
from ltx_trainer.mtllfm.localization import localize_laughter
from ltx_trainer.mtllfm.pooling import temporal_softmax_pool


def framework_card(cfg: MTLLFMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MTLLFMConfig()
    return {
        "name": "MTLLFM",
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "task": "Weakly-supervised temporal laughter localization",
        "encoders": ["HuBERT-Large (frozen)", "MAE (frozen)"],
        "components": [
            "Temporal softmax pooling",
            "Adaptive modality gating",
            "Peak-expansion localization",
        ],
        "datasets": ["UR-FUNNY-Temporal", "SMILE-Temporal", "SportsPress (proprietary)"],
    }


def table_dataset_stats() -> dict[str, dict[str, Any]]:
    """Table 1 — UR-FUNNY-Temporal and SMILE-Temporal statistics."""
    return {
        "UR-FUNNY-Temporal": {
            "total_videos": 10166,
            "total_hours": 72.9,
            "videos_with_laughter": 2369,
            "laughter_events": 3385,
            "mean_duration_s": 1.70,
            "acoustic_pct": 79.4,
            "audience_pct": 81.6,
        },
        "SMILE-Temporal": {
            "total_videos": 887,
            "total_hours": 5.9,
            "videos_with_laughter": 589,
            "laughter_events": 1560,
            "mean_duration_s": 2.16,
            "acoustic_pct": 92.5,
            "audience_pct": 93.7,
        },
    }


def table_foundation_comparison() -> dict[str, dict[str, dict[str, float]]]:
    """Table 2 — classification F1 and localization @ IoU=0.5."""
    return {
        "SportsPress": {
            "Qwen2 Audio 7B": {"cls_f1": 0.982, "loc_at_05": 0.028, "iou": 0.059},
            "Qwen2.5 Omni 7B": {"cls_f1": 0.997, "loc_at_05": 0.208, "iou": 0.301},
            "Gemini 3 Flash": {"cls_f1": 0.885, "loc_at_05": 0.542, "iou": 0.546},
            "MTLLFM": {"cls_f1": 0.990, "loc_at_05": 0.681, "iou": 0.580},
        },
        "UR-Funny": {
            "Qwen2 Audio 7B": {"cls_f1": 0.466, "loc_at_05": 0.137, "iou": 0.172},
            "Qwen2.5 Omni 7B": {"cls_f1": 0.773, "loc_at_05": 0.218, "iou": 0.267},
            "Gemini 3 Flash": {"cls_f1": 0.775, "loc_at_05": 0.393, "iou": 0.405},
            "MTLLFM": {"cls_f1": 0.849, "loc_at_05": 0.497, "iou": 0.466},
        },
        "SMILE": {
            "Qwen2 Audio 7B": {"cls_f1": 0.335, "loc_at_05": 0.138, "iou": 0.204},
            "Qwen2.5 Omni 7B": {"cls_f1": 0.783, "loc_at_05": 0.315, "iou": 0.361},
            "Gemini 3 Flash": {"cls_f1": 0.724, "loc_at_05": 0.579, "iou": 0.540},
            "MTLLFM": {"cls_f1": 0.803, "loc_at_05": 0.567, "iou": 0.511},
        },
    }


def table_ablation_sportspress() -> dict[str, dict[str, float]]:
    """Table 3 — SportsPress ablations."""
    return {
        "Audio Only": {"f1": 0.979, "p_at_05": 0.604, "iou": 0.552},
        "Vision Only": {"f1": 0.675, "p_at_05": 0.257, "iou": 0.322},
        "w/o Tanh Gating": {"f1": 0.979, "p_at_05": 0.639, "iou": 0.562},
        "Mean Pool": {"f1": 0.968, "p_at_05": 0.160, "iou": 0.347},
        "Max Pool": {"f1": 0.946, "p_at_05": 0.160, "iou": 0.347},
        "Self-Attention Pool": {"f1": 0.653, "p_at_05": 0.188, "iou": 0.274},
        "Concat (no gate)": {"f1": 0.976, "p_at_05": 0.618, "iou": 0.535},
        "Sigmoid Gate": {"f1": 0.986, "p_at_05": 0.625, "iou": 0.555},
        "Cross-Attention Fusion": {"f1": 0.677, "p_at_05": 0.248, "iou": 0.324},
        "Full Model (Ours)": {"f1": 0.990, "p_at_05": 0.681, "iou": 0.580},
    }


def table_downstream_reasoning() -> dict[str, dict[str, float]]:
    """Table 4 — Video Laugh Reasoning with temporal tags."""
    return {
        "GPT-3.5 Baseline": {"bleu4": 0.148, "meteor": 0.321, "bertscore": 0.393, "cider": 0.262},
        "GPT-4o Baseline": {"bleu4": 0.164, "meteor": 0.365, "bertscore": 0.400, "cider": 0.516},
        "GPT-3.5 + Tags (Ours)": {"bleu4": 0.235, "meteor": 0.423, "bertscore": 0.459, "cider": 0.858},
    }


def training_step_demo(cfg: MTLLFMConfig | None = None) -> dict[str, float]:
    """Smoke: temporal pool → gating → localization on synthetic peaks."""
    cfg = cfg or MTLLFMConfig()
    torch.manual_seed(25409)
    t_a, t_v = 50, 50
    d = cfg.hidden_dim

    audio = torch.randn(t_a, d) * 0.1
    visual = torch.randn(t_v, d) * 0.1
    # inject laughter peak mid-segment
    audio[t_a // 2] += 3.0
    visual[t_v // 2] += 2.5

    f_a, alpha_a = temporal_softmax_pool(audio)
    f_v, alpha_v = temporal_softmax_pool(visual)
    f_fused, w_a, w_v = adaptive_modality_gating(f_a, f_v)
    loc = localize_laughter(
        alpha_a,
        alpha_v,
        w_a=w_a,
        w_v=w_v,
        num_bins=cfg.localization_bins,
        temperature=cfg.sharpen_temperature,
        segment_seconds=cfg.segment_seconds,
    )

    return {
        "fused_norm": float(f_fused.norm().detach()),
        "w_audio": w_a,
        "w_visual": w_v,
        "peak_attention": loc["peak_attention"],
        "loc_start_sec": loc["start_sec"],
        "loc_end_sec": loc["end_sec"],
        "alpha_audio_peak_idx": float(alpha_a.argmax().item()),
    }


def evaluation_demo(cfg: MTLLFMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MTLLFMConfig()
    step = training_step_demo(cfg)
    cmp = table_foundation_comparison()
    abl = table_ablation_sportspress()
    reason = table_downstream_reasoning()

    cider_gain = (
        reason["GPT-3.5 + Tags (Ours)"]["cider"] / reason["GPT-3.5 Baseline"]["cider"] - 1.0
    ) * 100.0

    return {
        **step,
        "sportspress_beats_gemini_loc": cmp["SportsPress"]["MTLLFM"]["loc_at_05"]
        > cmp["SportsPress"]["Gemini 3 Flash"]["loc_at_05"],
        "full_model_best_ablation": abl["Full Model (Ours)"]["p_at_05"]
        >= max(v["p_at_05"] for v in abl.values()),
        "gpt35_cider_gain_pct": cider_gain,
        "gpt35_tags_beats_gpt4o_baseline": reason["GPT-3.5 + Tags (Ours)"]["cider"]
        > reason["GPT-4o Baseline"]["cider"],
        "total_annotated_videos": 10166 + 887,
    }
