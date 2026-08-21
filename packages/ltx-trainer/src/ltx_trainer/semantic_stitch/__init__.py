"""SemanticStitch: foreground-aware seam carving."""

from ltx_trainer.semantic_stitch.benchmarks import benchmarks_bundle
from ltx_trainer.semantic_stitch.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, SemanticStitchConfig
from ltx_trainer.semantic_stitch.integration import proceduralsky_card, proceduralsky_pipeline_plan
from ltx_trainer.semantic_stitch.ltx_plan import gopex_env_exports, ltx_training_plan
from ltx_trainer.semantic_stitch.mock import evaluation_smoke
from ltx_trainer.semantic_stitch.paper import framework_card
from ltx_trainer.semantic_stitch.pipeline import run_proceduralsky_bridge, run_stitch_smoke

__all__ = [
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "SemanticStitchConfig",
    "benchmarks_bundle",
    "evaluation_smoke",
    "framework_card",
    "proceduralsky_card",
    "proceduralsky_pipeline_plan",
    "gopex_env_exports",
    "ltx_training_plan",
    "run_proceduralsky_bridge",
    "run_stitch_smoke",
]
