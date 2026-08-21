"""SVHighlights framework card, paper tables, and demos (arXiv:2606.06926)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.svhighlights.config import SportCategory, SvHighlightsConfig
from ltx_trainer.svhighlights.layout import LIMITATIONS


def framework_card(cfg: SvHighlightsConfig | None = None) -> dict[str, Any]:
    c = cfg or SvHighlightsConfig()
    return {
        "name": "SVHighlights — extremely long sport video highlight detection",
        "paper": c.paper_arxiv,
        "doi": c.paper_doi,
        "url": c.paper_url,
        "authors": "Donggyu Lee, Youngbin Ki, Jeonghun Kang, Taehwan Kim (UNIST)",
        "venue": "KDD 2026",
        "task": "Highlight detection in hour-long sports broadcasts via official highlight alignment",
        "baseline": "TF-SELECTOR — training-free context-aware segments + VLM caption + LLM saliency",
        "dataset": {
            "n_videos": c.n_videos,
            "total_hours": c.total_hours,
            "avg_duration_min": c.avg_duration_min,
            "sports": [s.value for s in c.sports],
            "videos_per_sport": c.videos_per_sport,
        },
        "pipeline": [
            "Collect full-length + official highlight pairs (YouTube)",
            "Manual game-boundary trim; PSNR alignment @ 144p + temporal τ post-process",
            "PSNR<20 filter + lightweight manual grid verification",
            "2 s clip labels (50% overlap with 1 s highlight windows)",
            "TF-SELECTOR: shot merge → VLM caption → LLM 0–5 saliency → clip weighted sum",
        ],
        "metrics": ["mAP", "HIT@1", "HIT@K", "IoU", "window_F1"],
        "defaults": {k: v for k, v in c.__dict__.items() if not k.startswith("_")},
        "limitations": LIMITATIONS,
    }


def table_i_benchmarks() -> list[dict[str, str | float]]:
    """Table 1 — highlight detection dataset statistics."""
    rows = [
        ("YouTube HL", 712, 23.8, 2.0, "Auto/Human"),
        ("TVSum", 50, 3.5, 4.2, "Human"),
        ("QVHighlights", 10148, 422.8, 2.5, "Human"),
        ("Mr.HiSum", 31892, 1788.0, 3.4, "Auto"),
        ("SVHighlights", 320, 640.18, 120.0, "Auto(Human)"),
    ]
    cols = ("benchmark", "n_videos", "duration_h", "avg_min", "labels")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def table_iii_alignment_quality() -> list[dict[str, str | float]]:
    """Table 3 — per-sport alignment quality after filtering."""
    rows = [
        ("Amer. Football", 26.64, 0.872, 0.969, 93.7),
        ("Baseball", 26.28, 0.835, 0.943, 85.8),
        ("Basketball", 26.31, 0.868, 0.957, 94.7),
        ("Ice Hockey", 26.63, 0.856, 0.973, 94.6),
        ("Racing", 27.99, 0.873, 0.939, 92.4),
        ("Rugby", 27.43, 0.865, 0.941, 87.2),
        ("Soccer", 26.43, 0.877, 0.953, 90.0),
        ("Volleyball", 26.23, 0.874, 0.961, 95.3),
        ("All", 26.74, 0.865, 0.955, 91.7),
    ]
    cols = ("category", "psnr", "ssim", "clip_sim", "remain_rate_pct")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def table_vii_main_results() -> list[dict[str, str | float]]:
    """Table 7 — zero-shot performance on SVHighlights."""
    rows = [
        ("Moment-DETR", "V", "Clip", 9.16, 6.25, 7.61, 4.05),
        ("UMT", "V+A", "Clip", 10.97, 13.44, 11.76, 6.39),
        ("QD-DETR", "V", "Clip", 10.87, 7.50, 10.70, 6.00),
        ("MH-DETR", "V", "Clip", 8.49, 3.44, 4.52, 2.42),
        ("UniVTG", "V", "Clip", 8.76, 2.19, 5.68, 2.97),
        ("TR-DETR", "V", "Clip", 12.29, 23.44, 12.84, 7.05),
        ("CG-DETR", "V", "Clip", 10.43, 9.38, 9.95, 5.52),
        ("SL-Module", "V", "Segment", 8.82, 5.62, 5.99, 3.14),
        ("VTG-LLM", "V", "Clip", 11.64, 23.44, 9.40, 7.63),
        ("TimeChat", "V", "Clip", 12.40, 14.69, 9.42, 7.63),
        ("TRACE", "V", "Clip", 23.14, 23.12, 9.38, 7.63),
        ("TF-SELECTOR", "V+A", "Segment", 12.81, 26.56, 16.90, 10.58),
    ]
    cols = ("method", "input", "scoring", "mAP", "hit_at_1", "hit_at_k", "iou")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def table_viii_vlm_ablation() -> list[dict[str, str | float]]:
    """Table 8 — captioner (VLM) ablation."""
    rows = [
        ("LLaVA-OV-7B", 12.42, 25.31, 15.28, 9.86),
        ("Qwen2.5-VL-7B", 13.54, 28.12, 16.72, 10.09),
        ("InternVL2.5-8B", 12.81, 26.56, 16.90, 10.58),
    ]
    cols = ("captioner", "mAP", "hit_at_1", "hit_at_k", "iou")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def table_ix_llm_ablation() -> list[dict[str, str | float]]:
    """Table 9 — LLM ablation."""
    rows = [
        ("Llama2-7B", 9.59, 13.75, 11.34, 7.68),
        ("Qwen2.5-7B", 9.90, 21.88, 13.80, 9.53),
        ("Mistral-7B", 11.05, 21.25, 15.14, 10.02),
        ("Llama3-8B", 12.81, 26.56, 16.90, 10.58),
    ]
    cols = ("llm", "mAP", "hit_at_1", "hit_at_k", "iou")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def table_x_modality_ablation() -> list[dict[str, str | float]]:
    """Table 10 — input modality ablation."""
    rows = [
        ("(C) → S", 10.54, 15.00, 12.54, 8.64),
        ("(C, A) → S", 10.29, 16.25, 12.38, 9.23),
        ("(C, T) → S", 12.18, 21.25, 16.21, 11.11),
        ("(C, T, A) → S", 12.81, 26.56, 16.90, 10.58),
    ]
    cols = ("modality", "mAP", "hit_at_1", "hit_at_k", "iou")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def table_xii_window_f1() -> list[dict[str, str | float]]:
    """Table 12 — window-based F1 (Appendix D)."""
    rows = [
        ("UMT", 10.97, 13.44, 11.76, 6.39, 29.91),
        ("TR-DETR", 12.29, 23.44, 12.84, 7.05, 23.84),
        ("TRACE", 23.14, 23.12, 9.38, 7.63, 17.61),
        ("TF-SELECTOR", 12.81, 26.56, 16.90, 10.58, 27.38),
    ]
    cols = ("method", "mAP", "hit_at_1", "hit_at_k", "iou", "window_f1")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def table_tau_ablation() -> list[dict[str, str | float]]:
    """Table 5 — post-processing threshold τ ablation."""
    return [
        {"tau": 1, "remain_rate_pct": 89.78, "f_star_ratio_pct": 72.3, "f_plus_ratio_pct": 27.7},
        {"tau": 5, "remain_rate_pct": 88.16, "f_star_ratio_pct": 51.7, "f_plus_ratio_pct": 48.3},
        {"tau": 10, "remain_rate_pct": 80.36, "f_star_ratio_pct": 27.4, "f_plus_ratio_pct": 72.6},
    ]


def trimming_cues() -> list[dict[str, str]]:
    """Table 11 — per-sport game start/end cues."""
    return [
        {"sport": "Amer. Football", "game_start": "Opening whistle", "game_end": "Final whistle"},
        {"sport": "Baseball", "game_start": "Just before first pitch", "game_end": "Just after last out"},
        {"sport": "Basketball", "game_start": "Opening whistle", "game_end": "Final buzzer"},
        {"sport": "Ice Hockey", "game_start": "Opening faceoff", "game_end": "Final buzzer"},
        {"sport": "Racing", "game_start": "Starting signal", "game_end": "Winner's checkered flag"},
        {"sport": "Rugby", "game_start": "Kickoff whistle", "game_end": "Final whistle"},
        {"sport": "Soccer", "game_start": "Kickoff whistle", "game_end": "Final whistle"},
        {"sport": "Volleyball", "game_start": "Opening whistle", "game_end": "Final set match point"},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i": table_i_benchmarks(),
        "table_iii_alignment": table_iii_alignment_quality(),
        "table_vii": table_vii_main_results(),
        "table_viii_vlm": table_viii_vlm_ablation(),
        "table_ix_llm": table_ix_llm_ablation(),
        "table_x_modality": table_x_modality_ablation(),
        "table_xii_window_f1": table_xii_window_f1(),
        "table_tau_ablation": table_tau_ablation(),
        "trimming_cues": trimming_cues(),
        "sports": [s.value for s in SportCategory],
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    from ltx_trainer.svhighlights.mock import (
        run_alignment_smoke,
        run_label_smoke,
        run_metrics_smoke,
        run_tf_selector_smoke,
    )

    return {
        "framework": framework_card(),
        "alignment": run_alignment_smoke(seed=seed),
        "labeling": run_label_smoke(seed=seed + 1),
        "tf_selector": run_tf_selector_smoke(seed=seed + 2),
        "metrics": run_metrics_smoke(seed=seed + 3),
        "benchmarks": {
            "svhighlights_avg_min": next(r["avg_min"] for r in table_i_benchmarks() if r["benchmark"] == "SVHighlights"),
            "tf_selector_hit_at_1": next(r["hit_at_1"] for r in table_vii_main_results() if r["method"] == "TF-SELECTOR"),
        },
    }
