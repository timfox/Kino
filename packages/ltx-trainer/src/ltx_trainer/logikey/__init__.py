"""LogiKEy logical pluralism stub (arXiv:2605.27246)."""

from ltx_trainer.logikey.config import LogikeyConfig
from ltx_trainer.logikey.godel import distinct_positive_from_entity_count
from ltx_trainer.logikey.homl import KripkeFrame, mvalid
from ltx_trainer.logikey.layout import LIMITATIONS, LOGIKEY_LAYERS, PIPELINE_STAGES
from ltx_trainer.logikey.mock import evaluation_smoke
from ltx_trainer.logikey.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.logikey.tables import headline_results, logikey_layer_table

__all__ = [
    "LIMITATIONS",
    "LOGIKEY_LAYERS",
    "PIPELINE_STAGES",
    "KripkeFrame",
    "LogikeyConfig",
    "benchmarks_bundle",
    "distinct_positive_from_entity_count",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "logikey_layer_table",
    "mvalid",
]
