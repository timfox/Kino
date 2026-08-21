"""South Asian music LLM understanding + generation benchmark (Kader et al., arXiv:2606.05522)."""

from ltx_trainer.sa_music.benchmark import (
    SUBTASKS,
    accuracy,
    extract_mcq_answer,
    total_questions,
)
from ltx_trainer.sa_music.config import SaMusicConfig
from ltx_trainer.sa_music.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.sa_music.generation import PROMPT_LEVELS, build_prompt
from ltx_trainer.sa_music.metrics import (
    kl_divergence,
    pearson_r,
    pitch_entropy,
    pitch_histogram,
    scale_adherence,
)
from ltx_trainer.sa_music.mock import evaluation_smoke
from ltx_trainer.sa_music.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table2_human_eval,
    table6_automatic_l3,
    table_understanding_anchors,
)

__all__ = [
    "PROMPT_LEVELS",
    "SUBTASKS",
    "SaMusicConfig",
    "accuracy",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "build_prompt",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "extract_mcq_answer",
    "framework_card",
    "headline_results",
    "kl_divergence",
    "pearson_r",
    "pipeline_demo",
    "pipeline_demo_export",
    "pitch_entropy",
    "pitch_histogram",
    "scale_adherence",
    "table2_human_eval",
    "table6_automatic_l3",
    "table_understanding_anchors",
    "total_questions",
]

from ltx_trainer.sa_music.fold import annotate_audio_save_data  # noqa: E402
