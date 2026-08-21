"""CADENet: async dual-stream adverse-weather perception (Khairy & Elias arXiv:2605.19837)."""

from ltx_trainer.cadenet.metrics import delta_f1, detection_f1, recall_score
from ltx_trainer.cadenet.pipeline import CADENet, CADENetConfig, process_frame
from ltx_trainer.cadenet.schema import Detection, WeatherCondition

__all__ = [
    "CADENet",
    "CADENetConfig",
    "Detection",
    "WeatherCondition",
    "delta_f1",
    "detection_f1",
    "process_frame",
    "recall_score",
]
