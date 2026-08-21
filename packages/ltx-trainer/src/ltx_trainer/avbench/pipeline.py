"""AVBench framework card, paper tables, and smoke demos (arXiv:2605.24652)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.avbench.config import AVBenchConfig
from ltx_trainer.avbench.layout import LIMITATIONS
from ltx_trainer.avbench.mock import softmax_pair, toy_logits_aligned, toy_logits_misaligned
from ltx_trainer.avbench.scoring import audiobox_aesthetic_score, model_win_ratio, pearson_r, speech_content_score, yes_no_alignment_score
from ltx_trainer.avbench.taxonomy import SUITE_DIMENSIONS


def framework_card(cfg: AVBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AVBenchConfig()
    return {
        "name": "AVBench (Human-Aligned Automated Evaluation for Audio-Video Generation)",
        "paper": cfg.paper_arxiv,
        "project_page": cfg.project_page,
        "idea": (
            "Ten human-centric dimensions (cross-modal alignment + lip sync + six unimodal quality signals); "
            "specialized SFT evaluators on 300K hard-negative pairs output continuous Yes/(Yes+No) scores for RLHF-ready signals."
        ),
        "test_set": {
            "prompts_total": cfg.n_prompts_total,
            "normal_subset": cfg.n_prompts_normal,
            "hard_subset": cfg.n_prompts_hard,
            "isolation": "Hash dedup vs training clips; Qwen3-Omni annotations not reused as benchmark prompts",
        },
        "evaluator_training": {
            "source_clips": cfg.n_source_clips_openhumanvid,
            "clip_duration_s": list(cfg.clip_duration_seconds),
            "pairs_per_axis": cfg.n_pairs_per_axis,
            "pairs_total": cfg.n_pairs_total,
            "vt_av_backbone": cfg.backbone_vt_av,
            "at_backbone": cfg.backbone_at,
        },
        "suite_dimensions": [f"{a}: {b}" for a, b in SUITE_DIMENSIONS],
        "differentiable_signal": "Single-token Yes/No SFT; score = P(Yes)/(P(Yes)+P(No))",
    }


def table_benchmark_comparison() -> dict[str, dict[str, Any]]:
    """Table 1 — AVBench vs prior AV generation benchmarks."""
    return {
        "VBench": {
            "paradigm": "Zero-shot encoders",
            "training_data": "None",
            "human_centric": False,
            "automated_eval": True,
            "human_aligned": True,
            "differentiable_signal": False,
        },
        "T2AV-Compass": {
            "paradigm": "VQA-based",
            "training_data": "None",
            "human_centric": False,
            "automated_eval": False,
            "human_aligned": False,
            "differentiable_signal": False,
        },
        "VABench": {
            "paradigm": "VQA-based",
            "training_data": "None",
            "human_centric": False,
            "automated_eval": True,
            "human_aligned": False,
            "differentiable_signal": False,
        },
        "JointAVBench": {
            "paradigm": "VQA-based",
            "training_data": "None",
            "human_centric": False,
            "automated_eval": True,
            "human_aligned": False,
            "differentiable_signal": False,
        },
        "AVBench": {
            "paradigm": "Specialized SFT",
            "training_data": "300K (hard negatives)",
            "human_centric": True,
            "automated_eval": True,
            "human_aligned": True,
            "differentiable_signal": True,
        },
    }


def table_t2av_main_results() -> dict[str, dict[str, dict[str, float]]]:
    """Table 2 — SOTA T2AV models on AVBench (Normal / Hard splits)."""
    metrics = ("AV", "AT", "VT", "SyncNet", "SC", "DF_Arena", "NISQA", "Audiobox", "DOVERpp", "Aesthetic")
    normal = {
        "Sora 2": (0.8713, 0.8675, 0.7599, 4.9057, 87.8391, 0.4328, 2.3784, 3.1759, 60.0125, 4.0704),
        "Veo 3 Fast": (0.6924, 0.8300, 0.7235, 6.5943, 77.4950, 0.3043, 2.8191, 3.5877, 69.2275, 4.9967),
        "Wan 2.6": (0.8207, 0.8227, 0.7556, 4.5016, 91.5568, 0.0441, 3.0289, 3.9271, 71.6473, 4.7790),
        "Kling 2.6": (0.7626, 0.8061, 0.7501, 8.1027, 68.7844, 0.1665, 3.3141, 3.8082, 65.6786, 5.4885),
        "Seedance 1.5 Pro": (0.6536, 0.8554, 0.7363, 5.0146, 84.9268, 0.1602, 3.6411, 4.1686, 71.7205, 4.7373),
    }
    hard = {
        "Sora 2": (0.9320, 0.8575, 0.7190, 3.7932, 76.7905, 0.5498, 2.0564, 3.1339, 58.1538, 4.0434),
        "Veo 3 Fast": (0.7766, 0.8117, 0.6943, 3.4535, 70.3144, 0.3827, 2.3321, 3.6113, 67.0833, 5.1438),
        "Wan 2.6": (0.8780, 0.8418, 0.7482, 3.0488, 84.4512, 0.0498, 3.0726, 4.0924, 71.5229, 4.7721),
        "Kling 2.6": (0.8813, 0.7602, 0.7105, 3.9844, 69.0691, 0.1469, 3.2425, 3.8912, 62.9994, 5.5033),
        "Seedance 1.5 Pro": (0.7409, 0.8646, 0.7398, 3.3239, 80.8029, 0.2059, 3.4093, 4.1618, 69.4430, 4.7707),
    }
    def pack(row: tuple[float, ...]) -> dict[str, float]:
        return dict(zip(metrics, row, strict=True))

    return {"Normal": {k: pack(v) for k, v in normal.items()}, "Hard": {k: pack(v) for k, v in hard.items()}}


def table_evaluator_hard_negative_accuracy() -> dict[str, dict[str, float]]:
    """Figure 5a — hard-negative detection accuracy (paper-printed values; partial bar groups).

    AT/VT zero-shot rows and Qwen2-Audio AT baseline are explicit in Sec. 4.2; VT/AV rows
    list every numeric value shown in Fig. 5a without inventing extra baseline bars.
    """
    return {
        "Audio-Text": {
            "CLAP": 0.4888,
            "ViCLIP": 0.5494,
            "ImageBind": 0.4991,
            "Qwen2-Audio_no_ft": 0.2500,
            "Ours_SFT": 0.8437,
        },
        "Video-Text": {
            "CLAP": 0.4891,
            "ViCLIP": 0.3606,
            "Ours_SFT": 0.9144,
        },
        "Audio-Video": {
            "ImageBind": 0.4991,
            "Ours_SFT": 0.9817,
        },
    }


def table_human_alignment_pearson() -> dict[str, float]:
    """Sec. 4.3 — Pearson ρ between automated metric win ratios and human win ratios (selected dimensions)."""
    return {
        "AT_consistency": 0.9488,
        "VT_consistency": 0.9653,
        "speech_content": 0.9779,
        "DF_Arena_bonafide": 0.9668,
        "NISQA_MOS": 0.8012,
        "SyncNet": 0.8194,
    }


def table_human_2afc_accuracy() -> dict[str, float]:
    """Sec. 9.1 — instance-level agreement with human experts (seven objective dimensions; excerpt)."""
    return {
        "mean_over_dimensions_pct": 85.4,
        "speech_content_peak_pct": 98.1,
    }


def table_vt_consistency_instance_accuracy() -> dict[str, float]:
    """Fig. 14 — Video-Text consistency prediction accuracy."""
    return {"Ours_SFT_pct": 92.31, "Base_Qwen_pct": 47.44}


def evaluation_demo(cfg: AVBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AVBenchConfig()
    py, pn = softmax_pair(*toy_logits_aligned())
    py2, pn2 = softmax_pair(*toy_logits_misaligned())
    s_align = yes_no_alignment_score(py, pn)
    s_mis = yes_no_alignment_score(py2, pn2)
    # Win ratio smoke: three models with synthetic W/T/L
    wr = [model_win_ratio(12, 3, 5), model_win_ratio(8, 4, 8), model_win_ratio(15, 1, 4)]
    rho = pearson_r(wr, [0.55, 0.48, 0.62])
    main = table_t2av_main_results()
    return {
        "framework": framework_card(cfg),
        "scoring_smoke": {
            "aligned_S": s_align,
            "misaligned_S": s_mis,
            "speech_content_stub": speech_content_score(0.9, 0.85, 0.1),
            "audiobox_aesthetic_stub": audiobox_aesthetic_score(4.0, 4.2, 4.5, 2.0),
            "synthetic_win_ratios_pearson_r": rho,
        },
        "insights": {
            "VT_bottleneck": "VT scores lag AV across models; worsens on Hard split (paper Sec. 4.1).",
            "Kling_lip_sync_tradeoff": "Kling leads SyncNet but trails SC and DF_Arena vs Wan/Sora trade-offs.",
        },
        "paper_tables": {
            "benchmark_comparison": table_benchmark_comparison(),
            "t2av_main": main,
            "evaluator_accuracy": table_evaluator_hard_negative_accuracy(),
            "human_pearson": table_human_alignment_pearson(),
            "human_2afc": table_human_2afc_accuracy(),
            "vt_instance_accuracy": table_vt_consistency_instance_accuracy(),
        },
        "limitations": list(LIMITATIONS),
        "config": {
            "n_prompts_normal": cfg.n_prompts_normal,
            "n_prompts_hard": cfg.n_prompts_hard,
            "n_pairs_total": cfg.n_pairs_total,
        },
    }


def training_step_demo(cfg: AVBenchConfig | None = None) -> dict[str, Any]:
    """Reference hook: aggregate paper metrics for tooling parity with other trainer stubs."""
    cfg = cfg or AVBenchConfig()
    t2 = table_t2av_main_results()
    return {
        "loss_weights_note": "AVBench is an evaluator benchmark; no native training loss in this stub.",
        "normal_mean_VT": sum(t2["Normal"][m]["VT"] for m in t2["Normal"]) / len(t2["Normal"]),
        "hard_mean_VT": sum(t2["Hard"][m]["VT"] for m in t2["Hard"]) / len(t2["Hard"]),
        "evaluator_at_accuracy": table_evaluator_hard_negative_accuracy()["Audio-Text"]["Ours_SFT"],
    }
