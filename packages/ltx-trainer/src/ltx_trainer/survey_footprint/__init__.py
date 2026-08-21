"""Survey Footprint Explorer — MOC footprint visualisation stub (arXiv:2605.11099)."""

from ltx_trainer.survey_footprint.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.survey_footprint.catalogue import augment_catalogue, parse_catalogue_text
from ltx_trainer.survey_footprint.config import TOOL_URL, VERSION, SurveyFootprintConfig
from ltx_trainer.survey_footprint.moc_engine import (
    intersection_area_deg2,
    load_builtin_moc,
    filter_coordinates,
)
from ltx_trainer.survey_footprint.paper import evaluation_demo, framework_card
from ltx_trainer.survey_footprint.pipeline import evaluation_demo_run, use_case_a_overlap, use_case_b_catalogue_augment
from ltx_trainer.survey_footprint.surveys import BUILTIN_SURVEYS, SURVEY_BY_ID

__all__ = [
    "BUILTIN_SURVEYS",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "SURVEY_BY_ID",
    "TOOL_URL",
    "VERSION",
    "SurveyFootprintConfig",
    "augment_catalogue",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "filter_coordinates",
    "framework_card",
    "intersection_area_deg2",
    "load_builtin_moc",
    "parse_catalogue_text",
    "use_case_a_overlap",
    "use_case_b_catalogue_augment",
]
