"""RaFI — Ray/Work Forwarding Infrastructure (arXiv:2605.30294)."""

from ltx_trainer.rafi.config import RafiConfig
from ltx_trainer.rafi.host import HostContext
from ltx_trainer.rafi.mock import evaluation_smoke
from ltx_trainer.rafi.pipeline import dual_gpu_rank_map, evaluation_demo, framework_card

__all__ = [
    "RafiConfig",
    "HostContext",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "dual_gpu_rank_map",
]
