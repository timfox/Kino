"""KD-NVC — arXiv:2606.04595."""

from ltx_trainer.kd_nvc.config import KDNVCConfig, KD_NVC_S, KD_NVC_T, DistillConfig
from ltx_trainer.kd_nvc.pipeline import evaluation_demo, evaluation_smoke, framework_card

__all__ = [
    "KDNVCConfig",
    "KD_NVC_S",
    "KD_NVC_T",
    "DistillConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
]
