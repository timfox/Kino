"""IJCIT 2025 survey: facial recognition techniques (Bahjat)."""

from ltx_trainer.face_rec_survey.benchmarks import best_result_on, table_experimental_results
from ltx_trainer.face_rec_survey.challenges import (
    ChallengeFactor,
    EXTERNAL_FACTORS,
    INTERNAL_FACTORS,
    challenge_catalog,
    mitigation_strategies,
)
from ltx_trainer.face_rec_survey.config import (
    PAPER_DOI,
    PAPER_TITLE,
    PAPER_URL,
    FaceRecSurveyConfig,
)
from ltx_trainer.face_rec_survey.databases import table_face_databases
from ltx_trainer.face_rec_survey.integration import gopex_dataset_links, ltx_cast_consistency_notes
from ltx_trainer.face_rec_survey.methods import (
    RecognitionDomain,
    recognition_pipeline_stages,
    table_classical_methods,
    table_deep_methods,
)
from ltx_trainer.face_rec_survey.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    evaluation_smoke,
    framework_card,
)
from ltx_trainer.face_rec_survey.simulation import confounder_accuracy_report

__all__ = [
    "ChallengeFactor",
    "EXTERNAL_FACTORS",
    "INTERNAL_FACTORS",
    "FaceRecSurveyConfig",
    "PAPER_DOI",
    "PAPER_TITLE",
    "PAPER_URL",
    "RecognitionDomain",
    "benchmarks_bundle",
    "best_result_on",
    "challenge_catalog",
    "confounder_accuracy_report",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "gopex_dataset_links",
    "ltx_cast_consistency_notes",
    "mitigation_strategies",
    "recognition_pipeline_stages",
    "table_classical_methods",
    "table_deep_methods",
    "table_experimental_results",
    "table_face_databases",
]
