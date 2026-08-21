"""UWD lexicon evaluation — WNES / iWNES (arXiv:2606.06183)."""

from ltx_trainer.uwd_lex.config import UwdLexConfig
from ltx_trainer.uwd_lex.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.uwd_lex.fold import annotate_audio_save_data
from ltx_trainer.uwd_lex.metrics import (
    bitrate,
    d_pacc,
    evaluate_lexicon,
    f1_wnes,
    ipacc,
    iwnes,
    nes,
    normalized_edit_distance,
    pacc,
    toy_lexicon_demo,
    wnes,
)
from ltx_trainer.uwd_lex.mock import evaluation_smoke
from ltx_trainer.uwd_lex.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    fig4_real_world_anchors,
    fig6_synthetic_scores,
    framework_card,
    headline_results,
    pipeline_demo,
)

__all__ = [
    "UwdLexConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "bitrate",
    "d_pacc",
    "eval_smoke",
    "evaluate_lexicon",
    "evaluation_demo",
    "evaluation_smoke",
    "f1_wnes",
    "fig4_real_world_anchors",
    "fig6_synthetic_scores",
    "framework_card",
    "headline_results",
    "ipacc",
    "iwnes",
    "nes",
    "normalized_edit_distance",
    "pacc",
    "pipeline_demo",
    "pipeline_demo_export",
    "toy_lexicon_demo",
    "wnes",
]
