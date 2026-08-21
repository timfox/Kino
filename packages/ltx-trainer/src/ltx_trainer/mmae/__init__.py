"""MMAE — Massive Multitask Audio Editing benchmark (arXiv:2606.07229)."""

from ltx_trainer.mmae.benchmarks import benchmarks_bundle, paper_claims, table1_statistics, table2_main_results
from ltx_trainer.mmae.catalog import appendix_b_dog_extraction, catalog_summary, synthetic_catalog
from ltx_trainer.mmae.config import MMAEConfig, Complexity, Modality, RubricCategory
from ltx_trainer.mmae.evaluate import (
    evaluate_catalog,
    evaluate_hub,
    evaluate_samples,
    evaluate_submission,
    export_rubric_manifest,
    load_samples_for_eval,
    write_eval_report,
)
from ltx_trainer.mmae.editors import (
    editor_run_plan,
    editors_card,
    materialize_baseline,
    materialize_editor_submission,
    materialize_identity_baseline,
    materialize_noise_baseline,
)
from ltx_trainer.mmae.judger_backend import build_judger, parse_judger_choice_letter, probe_judger, resolve_judger_config
from ltx_trainer.mmae.layout import LIMITATIONS
from ltx_trainer.mmae.ltx_plan import gopex_env_snippet, ltx_integration_plan
from ltx_trainer.mmae.pipeline import evaluation_demo, evaluation_smoke, framework_card, paper_limitations
from ltx_trainer.mmae.sample import MMAESample, Rubric, load_sample_json
from ltx_trainer.mmae.simulation import evaluate_model_on_catalog, full_eval_demo
from ltx_trainer.mmae.submissions import build_predictions_json
from ltx_trainer.mmae.taxonomy import taxonomy_summary

__all__ = [
    "Complexity",
    "LIMITATIONS",
    "MMAEConfig",
    "MMAESample",
    "Modality",
    "Rubric",
    "RubricCategory",
    "appendix_b_dog_extraction",
    "build_predictions_json",
    "catalog_summary",
    "editor_run_plan",
    "editors_card",
    "evaluate_catalog",
    "evaluate_hub",
    "evaluate_model_on_catalog",
    "evaluate_samples",
    "evaluate_submission",
    "export_rubric_manifest",
    "load_samples_for_eval",
    "materialize_baseline",
    "materialize_editor_submission",
    "materialize_identity_baseline",
    "materialize_noise_baseline",
    "probe_judger",
    "resolve_judger_config",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "full_eval_demo",
    "gopex_env_snippet",
    "load_sample_json",
    "ltx_integration_plan",
    "paper_claims",
    "paper_limitations",
    "synthetic_catalog",
    "table1_statistics",
    "table2_main_results",
    "taxonomy_summary",
    "write_eval_report",
]
