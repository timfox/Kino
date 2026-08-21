"""PiD — Pixel diffusion Decoder (NVIDIA, arXiv:2605.23902)."""

from ltx_trainer.pid.config import PiDConfig
from ltx_trainer.pid.core import pid_decode_step
from ltx_trainer.pid.decode import PiDDecoder, training_loss
from ltx_trainer.pid.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)

__all__ = [
    "PiDConfig",
    "PiDDecoder",
    "pid_decode_step",
    "training_loss",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_card",
]
