"""torch-webgpu — WebGPU dispatch overhead characterization for LLM inference."""

from ltx_trainer.torch_webgpu.benchmarks import benchmarks_bundle
from ltx_trainer.torch_webgpu.config import TorchWebGPUConfig
from ltx_trainer.torch_webgpu.pipeline import evaluation_demo, evaluation_smoke, framework_card

__all__ = [
    "TorchWebGPUConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
]
