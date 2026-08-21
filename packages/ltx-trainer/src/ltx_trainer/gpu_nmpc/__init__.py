"""GPU-accelerated parametric NMPC (arXiv:2606.04725)."""

from ltx_trainer.gpu_nmpc.config import GpuNmpcConfig
from ltx_trainer.gpu_nmpc.pipeline import evaluation_demo, evaluation_smoke, framework_card

__all__ = ["GpuNmpcConfig", "evaluation_demo", "evaluation_smoke", "framework_card"]
