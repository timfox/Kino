"""TimesFM CPS attack detection — arXiv:2606.06347."""

from ltx_trainer.timesfm_attack.config import MassSpringConfig, TimesFMAttackConfig
from ltx_trainer.timesfm_attack.pipeline import evaluation_demo, evaluation_smoke, framework_card

__all__ = [
    "MassSpringConfig",
    "TimesFMAttackConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
]
