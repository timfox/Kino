"""DetectZoo — unified AI-generated content detection (arXiv:2606.04205)."""

from ltx_trainer.detectzoo.api import load_detector
from ltx_trainer.detectzoo.config import DetectZooConfig
from ltx_trainer.detectzoo.mock import evaluation_smoke
from ltx_trainer.detectzoo.pipeline import evaluation_demo, framework_card
from ltx_trainer.detectzoo.result import DetectionResult
from ltx_trainer.detectzoo.ltx_plan import ltx_integration_plan

__all__ = [
    "DetectZooConfig",
    "DetectionResult",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "load_detector",
    "ltx_integration_plan",
]
