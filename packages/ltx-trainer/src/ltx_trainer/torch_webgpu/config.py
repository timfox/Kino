"""torch-webgpu / WebGPU dispatch overhead configuration (Maczan arXiv:2604.02344)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2604.02344"
PAPER_TITLE = (
    "Characterizing WebGPU Dispatch Overhead for LLM Inference "
    "Across Four GPU Vendors, Three Backends, and Three Browsers"
)
PAPER_AUTHORS = "Jędrzej Maczan (Independent Researcher)"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_REPO = "https://github.com/jmaczan/torch-webgpu"
PAPER_VENUE = "Preprint (Feb 2026)"

# Qwen2.5 models (Sec. 3.3)
QWEN_05B_LAYERS = 24
QWEN_05B_HIDDEN = 896
QWEN_05B_INTER = 4864
QWEN_15B_LAYERS = 28
QWEN_15B_HIDDEN = 1536
QWEN_15B_INTER = 8960

# FX graph (Appendix B, Sec. 4.3)
FX_TOTAL_NODES = 1911
FX_COMPUTE_OPS = 876
FX_SHAPE_OPS = 241
DISPATCHES_UNFUSED = 876
DISPATCHES_FUSED = 564
DISPATCHES_SAVED_FUSION = DISPATCHES_UNFUSED - DISPATCHES_FUSED  # 312

# Overhead anchors (Sec. 4.4, Table 4)
TTFT_UNFUSED_MS = 71.4
TTFT_FUSED_MS = 41.6
PER_OPERATION_OVERHEAD_US = 95.0
PER_DISPATCH_DAWN_US = 23.8
PER_DISPATCH_WGPU_VULKAN_US = 35.8
SYNC_OVERHEAD_MS = 11.0

# Throughput anchors RTX 5090/Dawn fp32 (Table 2)
TOK_S_05B_FUSED = 21.0
TOK_S_15B_FUSED = 17.9
TOK_S_05B_UNFUSED = 13.5
CUDA_05B_FP16 = 185.5

# Kernel efficiency (Table 8)
RTX5090_FP32_PEAK_TFLOPS = 105.0
WGSL_THROUGHPUT_TFLOPS = 2.0  # measured baseline ~1–2% peak


@dataclass
class TorchWebGPUConfig:
    model: str = "Qwen2.5-0.5B-Instruct"
    dtype: str = "float32"
    batch_size: int = 1
    tokens_per_run: int = 50
    warmup_runs: int = 5
    timed_runs: int = 30
    per_dispatch_us: float = PER_DISPATCH_DAWN_US
    per_operation_us: float = PER_OPERATION_OVERHEAD_US
    dispatches_fused: int = DISPATCHES_FUSED
    dispatches_unfused: int = DISPATCHES_UNFUSED
