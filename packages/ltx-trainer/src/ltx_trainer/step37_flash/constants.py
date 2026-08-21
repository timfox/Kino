"""Step 3.7 Flash — constants from stepfun-ai/Step-3.7-Flash README."""

from __future__ import annotations

GITHUB_REPO = "https://github.com/stepfun-ai/Step-3.7-Flash"
MODEL_PAGE = "https://static.stepfun.com/blog/step-3.7-flash/"
LICENSE = "Apache-2.0"
API_MODEL_ID = "step-3.7-flash"

# Architecture
TOTAL_PARAMS_B = 198.0
LANGUAGE_PARAMS_B = 196.0
VISION_PARAMS_B = 1.8
ACTIVE_PARAMS_PER_TOKEN_B = 11.0
MAX_CONTEXT_TOKENS = 256_000
PEAK_THROUGHPUT_TPS = 400

REASONING_LEVELS = ("low", "medium", "high")

# Hugging Face weights
HF_WEIGHTS = {
    "bf16": "stepfun-ai/Step-3.7-Flash",
    "fp8": "stepfun-ai/Step-3.7-Flash-FP8",
    "nvfp4": "stepfun-ai/Step-3.7-Flash-NVFP4",
    "gguf": "stepfun-ai/Step-3.7-Flash-GGUF",
}

# API platforms (region-specific base_url)
API_REGIONS = {
    "global": {
        "platform": "https://platform.stepfun.ai",
        "base_url": "https://api.stepfun.ai/v1",
    },
    "china": {
        "platform": "https://platform.stepfun.com",
        "base_url": "https://api.stepfun.com/v1",
    },
}

# Pricing per million tokens (USD)
PRICING_PER_M = {
    "input_cache_miss": 0.20,
    "input_cache_hit": 0.04,
    "output": 1.15,
}

# Published benchmark scores (README §2)
BENCHMARK_SCORES = {
    "SimpleVQA_Search": 79.2,
    "V_star_Python": 95.3,
    "ClawEval_1_1": 67.1,
    "ClawEval_runner_up": 59.8,
    "Toolathlon": 49.5,
    "HLE_w_Tool": 48.1,
    "SWE_Bench_PRO": 56.3,
    "Terminal_Bench_2_1": 59.5,
    "GDPVal_AA": 45.8,
}

# Inference backends
INFERENCE_BACKENDS = ("vllm", "sglang", "transformers", "llama_cpp", "nim")

# Ecosystem partners (README §4)
AVAILABILITY_CHANNELS = (
    "StepFun Open Platform (global + China)",
    "OpenRouter",
    "NVIDIA NIM",
    "DeepInfra (coming)",
    "Fireworks AI (coming)",
    "Modal (coming)",
)

AGENT_PLATFORMS = ("Hermes Agent", "OpenClaw", "Kilo Code")
