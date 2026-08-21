"""Configuration for TextSculptor (Lin et al., arXiv:2605.21090)."""

from __future__ import annotations

from dataclasses import dataclass

TASK_TYPES: tuple[str, ...] = ("addition", "removal", "replacement", "hybrid")
VQ_CRITERIA: tuple[str, ...] = ("location", "style", "physical")


@dataclass
class TextSculptorConfig:
    """Defaults from paper Sec. 3–5."""

    # TextSculpt-Data
    t2i_candidates_m: float = 3.0
    t2i_verified_m: float = 1.2
    editing_pairs_m: float = 2.0
    total_samples_m: float = 3.2
    ocr_word_accuracy_threshold: float = 1.0  # perfect word-level
    # TextSculpt-Bench
    samples_per_task: int = 200
    pexels_images: int = 188
    poster_images: int = 168
    # Training (Sec. 5.1)
    lora_rank: int = 64
    learning_rate: float = 1e-4
    per_gpu_batch: int = 4
    grad_accum: int = 4
    num_gpus: int = 32
    # Table 1 — TextSculptor overall
    overall_ta: float = 0.60
    overall_vq: float = 0.68
    overall_bp: float = 0.78
    overall_avg: float = 0.69
    baseline_qwen_overall_avg: float = 0.65
    seedream45_overall_avg: float = 0.72
