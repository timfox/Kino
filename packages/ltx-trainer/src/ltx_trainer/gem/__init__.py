"""GEM geometric entropy mixing for LLM data curation (arXiv:2605.26121)."""

from ltx_trainer.gem.balance import empirical_mass, mixing_balance_grad, mixing_balance_value
from ltx_trainer.gem.config import GemConfig
from ltx_trainer.gem.gis import gis_score, top_gis_indices
from ltx_trainer.gem.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.gem.mm import GemFitResult, fit_gem, gem_objective
from ltx_trainer.gem.mock import evaluation_smoke
from ltx_trainer.gem.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.gem.tables import headline_results
from ltx_trainer.gem.vmf import kappa_from_r_bar, normalize_rows

__all__ = [
    "LIMITATIONS",
    "PIPELINE_STAGES",
    "GemConfig",
    "GemFitResult",
    "benchmarks_bundle",
    "empirical_mass",
    "evaluation_demo",
    "evaluation_smoke",
    "fit_gem",
    "framework_card",
    "gem_objective",
    "gis_score",
    "headline_results",
    "kappa_from_r_bar",
    "mixing_balance_grad",
    "mixing_balance_value",
    "normalize_rows",
    "top_gis_indices",
]
