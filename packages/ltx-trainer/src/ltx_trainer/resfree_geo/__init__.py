"""Resolution-free neural surrogates for geometric mapping (arXiv:2605.28551)."""

from ltx_trainer.resfree_geo.config import ResfreeGeoConfig
from ltx_trainer.resfree_geo.mock import evaluation_smoke
from ltx_trainer.resfree_geo.pipeline import benchmarks_bundle, evaluation_demo, framework_card

__all__ = [
    "ResfreeGeoConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "benchmarks_bundle",
]
