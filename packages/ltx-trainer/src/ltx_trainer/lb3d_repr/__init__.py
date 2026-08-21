"""CGF survey: learning-based 3D representations (Schockaert et al.; arXiv:2606.04871)."""

from ltx_trainer.lb3d_repr.benchmarks import (
    paper_anchors,
    table_3dgs_optimization,
    table_explicit_representations,
)
from ltx_trainer.lb3d_repr.config import (
    LB3DReprConfig,
    PAPER_ARXIV,
    PAPER_AUTHORS,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    DomainNature,
    RepresentationClass,
)
from ltx_trainer.lb3d_repr.integration import (
    gopex_stub_links,
    ltx_nvs_pipeline_notes,
    survey_to_gopex_mapping,
)
from ltx_trainer.lb3d_repr.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    evaluation_smoke,
    framework_card,
)
from ltx_trainer.lb3d_repr.taxonomy import (
    UNIFIED_FORMULATION,
    acceleration_strategies,
    future_directions,
    landscape_nodes,
    survey_sections,
    taxonomy_tree,
)
from ltx_trainer.lb3d_repr.tradeoffs import representation_tradeoff_report

__all__ = [
    "DomainNature",
    "LB3DReprConfig",
    "PAPER_ARXIV",
    "PAPER_AUTHORS",
    "PAPER_TITLE",
    "PAPER_URL",
    "PAPER_VENUE",
    "RepresentationClass",
    "UNIFIED_FORMULATION",
    "acceleration_strategies",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "future_directions",
    "gopex_stub_links",
    "landscape_nodes",
    "ltx_nvs_pipeline_notes",
    "paper_anchors",
    "representation_tradeoff_report",
    "survey_sections",
    "survey_to_gopex_mapping",
    "table_3dgs_optimization",
    "table_explicit_representations",
    "taxonomy_tree",
]
