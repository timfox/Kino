"""Framework card, tables, and demo entry points."""

from __future__ import annotations

from typing import Any

from ltx_trainer.face_rec_survey.benchmarks import table_experimental_results
from ltx_trainer.face_rec_survey.challenges import challenge_catalog
from ltx_trainer.face_rec_survey.config import (
    PAPER_AUTHOR,
    PAPER_DOI,
    PAPER_JOURNAL,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_YEAR,
    FaceRecSurveyConfig,
)
from ltx_trainer.face_rec_survey.databases import table_face_databases
from ltx_trainer.face_rec_survey.integration import gopex_dataset_links, ltx_cast_consistency_notes
from ltx_trainer.face_rec_survey.methods import (
    DETECTORS,
    FEATURE_EXTRACTORS,
    RecognitionDomain,
    recognition_pipeline_stages,
    table_classical_methods,
    table_deep_methods,
)
from ltx_trainer.face_rec_survey.simulation import confounder_accuracy_report


def framework_card(cfg: FaceRecSurveyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FaceRecSurveyConfig()
    return {
        "paper": PAPER_DOI,
        "title": PAPER_TITLE,
        "author": PAPER_AUTHOR,
        "journal": PAPER_JOURNAL,
        "year": PAPER_YEAR,
        "url": PAPER_URL,
        "pipeline_stages": recognition_pipeline_stages(),
        "domains": [d.value for d in RecognitionDomain],
        "classical_methods": [m["name"] for m in table_classical_methods()],
        "deep_methods": [m["name"] for m in table_deep_methods()],
        "detectors": DETECTORS,
        "feature_extractors": FEATURE_EXTRACTORS,
        "challenges": [c["factor"] for c in challenge_catalog()],
        "gopex_datasets": gopex_dataset_links(),
        "config": cfg.__dict__,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_databases": table_face_databases(),
        "table4_experimental": table_experimental_results(),
        "classical_methods": table_classical_methods(),
        "deep_methods": table_deep_methods(),
        "challenges": challenge_catalog(),
        "framework": framework_card(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    return {
        "simulation": confounder_accuracy_report(seed=seed),
        "cast_notes": ltx_cast_consistency_notes(),
        "best_orl": next(
            (r for r in table_experimental_results() if "ORL" in r["database"]),
            None,
        ),
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    sim = demo["simulation"]
    return {
        "paper": "face_rec_survey",
        "doi": PAPER_DOI,
        "baseline_accuracy": sim["baseline_accuracy"],
        "confounded": sim["confounded_accuracy"],
        "largest_drop_factor": sim["largest_drop_factor"],
        "database_count": len(table_face_databases()),
        "experimental_rows": len(table_experimental_results()),
    }
