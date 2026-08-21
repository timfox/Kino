"""Utility-Aware CLIP for product image generation (arXiv:2605.28733)."""

from ltx_trainer.uaclip.baselines import (
    TABLE1_AMAZON_DEMAND,
    TABLE2_AMAZON_EDITING,
    TABLE3_AIRBNB_DEMAND,
    TABLE4_AIRBNB_GENERATION,
    TABLE5_AIRBNB_EDITING,
    TABLE6_HUMAN_AMAZON,
    TABLE7_REGRESSION_AMAZON,
    TABLE8_AIRBNB_RATINGS,
    TABLE9_REGRESSION_AIRBNB,
)
from ltx_trainer.uaclip.compare import demand_at_optimal, score_candidate_batch
from ltx_trainer.uaclip.config import UAClipConfig
from ltx_trainer.uaclip.discrete_choice import conditional_logit_probabilities, infonce_from_utilities
from ltx_trainer.uaclip.demand import (
    Platform,
    airbnb_demand_score,
    amazon_demand_score,
    amazon_optimal_attrs,
    utility_regularized_similarity,
    visual_utility,
)
from ltx_trainer.uaclip.generator import CandidateImage, score_candidates, select_best
from ltx_trainer.uaclip.infonce import bidirectional_infonce, build_score_matrix, infonce_text_to_image
from ltx_trainer.uaclip.mock import evaluation_smoke
from ltx_trainer.uaclip.occlusion import patch_occlusion_sensitivity
from ltx_trainer.uaclip.pipeline import benchmarks_bundle, evaluation_demo, framework_card, knowledge_card
from ltx_trainer.uaclip.similarity import clip_similarity, utility_aware_score
from ltx_trainer.uaclip.theory import utility_aware_infonce_bound
from ltx_trainer.uaclip.training import UAClipTrainReport, demo_train as demo_train_uaclip, train_utility_aware_step
from ltx_trainer.uaclip.visual_attrs import VisualAttributes

__all__ = [
    "TABLE1_AMAZON_DEMAND",
    "TABLE2_AMAZON_EDITING",
    "TABLE3_AIRBNB_DEMAND",
    "TABLE4_AIRBNB_GENERATION",
    "TABLE5_AIRBNB_EDITING",
    "TABLE6_HUMAN_AMAZON",
    "TABLE7_REGRESSION_AMAZON",
    "TABLE8_AIRBNB_RATINGS",
    "TABLE9_REGRESSION_AIRBNB",
    "Platform",
    "UAClipConfig",
    "VisualAttributes",
    "airbnb_demand_score",
    "amazon_demand_score",
    "amazon_optimal_attrs",
    "UAClipTrainReport",
    "benchmarks_bundle",
    "demand_at_optimal",
    "score_candidate_batch",
    "demo_train_uaclip",
    "train_utility_aware_step",
    "bidirectional_infonce",
    "build_score_matrix",
    "CandidateImage",
    "clip_similarity",
    "conditional_logit_probabilities",
    "infonce_from_utilities",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "infonce_text_to_image",
    "knowledge_card",
    "patch_occlusion_sensitivity",
    "score_candidates",
    "select_best",
    "utility_aware_infonce_bound",
    "utility_aware_score",
    "utility_regularized_similarity",
    "visual_utility",
]
