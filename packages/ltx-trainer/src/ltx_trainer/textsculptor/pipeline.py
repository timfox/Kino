"""TextSculptor framework card, benchmark tables, and demos."""

from __future__ import annotations

from typing import Any

from ltx_trainer.textsculptor.config import TASK_TYPES, TextSculptorConfig
from ltx_trainer.textsculptor.data import bench_card, dataset_card
from ltx_trainer.textsculptor.metrics import average_score, text_accuracy, visual_quality_score


def framework_card(cfg: TextSculptorConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TextSculptorConfig()
    return {
        "name": "TextSculptor",
        "paper": "arXiv:2605.21090",
        "title": "Training and Benchmarking Scene Text Editing",
        "base_model": "Qwen-Image-Edit-2511",
        "dataset": dataset_card(),
        "benchmark": bench_card(),
        "task_types": list(TASK_TYPES),
        "training": {
            "lora_rank": cfg.lora_rank,
            "lr": cfg.learning_rate,
            "effective_batch": cfg.per_gpu_batch * cfg.grad_accum * cfg.num_gpus,
        },
        "code": "https://github.com/linyiheng123/TextSculptor",
    }


def table_main_results() -> dict[str, dict[str, float]]:
    """Table 1 — selected models (overall column)."""
    rows = {
        "gemini_25_flash": {"ta": 0.63, "vq": 0.69, "bp": 0.72, "avg": 0.68},
        "seedream45": {"ta": 0.76, "vq": 0.76, "bp": 0.63, "avg": 0.72},
        "qwen_image_edit_2511": {"ta": 0.55, "vq": 0.63, "bp": 0.76, "avg": 0.65},
        "firered_image_edit": {"ta": 0.62, "vq": 0.70, "bp": 0.73, "avg": 0.68},
        "textsculptor_ours": {"ta": 0.60, "vq": 0.68, "bp": 0.78, "avg": 0.69},
        "omnigen2": {"ta": 0.21, "vq": 0.28, "bp": 0.67, "avg": 0.39},
    }
    return rows


def table_per_task_textsculptor() -> dict[str, dict[str, float]]:
    """Table 1 — TextSculptor per task (TA/VQ/BP)."""
    return {
        "addition": {"ta": 0.33, "vq": 0.44, "bp": 0.79},
        "removal": {"ta": 0.70, "vq": 0.82, "bp": 0.77},
        "replacement": {"ta": 0.74, "vq": 0.75, "bp": 0.77},
        "hybrid": {"ta": 0.62, "vq": 0.72, "bp": 0.79},
    }


def table_ablation() -> dict[str, dict[str, float]]:
    """Table 2 — data ablations."""
    return {
        "baseline": {"ta": 0.55, "vq": 0.63, "bp": 0.76, "avg": 0.65},
        "full": {"ta": 0.60, "vq": 0.68, "bp": 0.78, "avg": 0.69},
        "wo_distraction": {"ta": 0.57, "vq": 0.60, "bp": 0.79, "avg": 0.65},
        "wo_t2i": {"ta": 0.56, "vq": 0.61, "bp": 0.80, "avg": 0.66},
    }


def evaluation_demo() -> dict[str, float]:
    """Smoke evaluation on synthetic strings."""
    expected = "Langoosh Pepi RACING THE TRAIN COMING SOON IN THEATERS 2026"
    observed = "Langoosh Pepi RACING FROM THE TRAIN IN THEATERS 2026"
    ta = text_accuracy(expected, observed, n_edit_words=4)
    vq = visual_quality_score(True, True, False)
    bp = 0.86
    return {
        "text_accuracy": ta,
        "visual_quality": vq,
        "background_preservation": bp,
        "average": average_score(ta, vq, bp),
    }
