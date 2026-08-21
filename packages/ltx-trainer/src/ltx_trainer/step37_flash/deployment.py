"""Local deployment command templates — vLLM, SGLang, llama.cpp."""

from __future__ import annotations

from typing import Any

from ltx_trainer.step37_flash.constants import HF_WEIGHTS, INFERENCE_BACKENDS


def vllm_serve_command(
    *,
    precision: str = "fp8",
    model_path: str | None = None,
    tensor_parallel: int = 8,
) -> str:
    """Shell command from README §6.1."""
    precision = precision.lower()
    model = model_path or HF_WEIGHTS.get(precision, HF_WEIGHTS["fp8"])
    if precision == "nvfp4":
        return (
            f"python3 -m vllm.entrypoints.openai.api_server "
            f"--host 0.0.0.0 --port 8000 "
            f"--model {model} --served-model-name step3p7 "
            f"--tensor-parallel-size 4 --gpu-memory-utilization 0.9 "
            f"--enable-expert-parallel --trust-remote-code "
            f"--quantization modelopt --kv-cache-dtype fp8 --max-model-len 8192 "
            f"--reasoning-parser step3p5 --enable-auto-tool-choice "
            f"--tool-call-parser step3p5 --async-scheduling"
        )
    served = "step3p7-flash" if precision == "fp8" else "step3p7-flash-bf16"
    return (
        f"vllm serve {model} "
        f"--served-model-name {served} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--enable-expert-parallel --disable-cascade-attn "
        f"--reasoning-parser step3p5 --enable-auto-tool-choice "
        f"--tool-call-parser step3p5 "
        f"--speculative_config '{{\"method\": \"mtp\", \"num_speculative_tokens\": 3}}' "
        f"--trust-remote-code"
    )


def sglang_serve_command(
    *,
    precision: str = "bf16",
    tensor_parallel: int = 8,
    port: int = 8000,
) -> str:
    """Shell command from README §6.2."""
    precision = precision.lower()
    model = HF_WEIGHTS.get(precision, HF_WEIGHTS["bf16"])
    ep = "--ep 4 " if precision == "fp8" else ""
    extra = ""
    if precision == "nvfp4":
        extra = (
            "--moe-runner-backend flashinfer_trtllm "
            "--kv-cache-dtype fp8_e4m3 --quantization modelopt_fp4 "
            "--attention-backend trtllm_mha "
        )
    return (
        f"sglang serve --model-path {model} "
        f"--tp {tensor_parallel} {ep}"
        f"--reasoning-parser step3p5 --tool-call-parser step3p5 "
        f"--enable-multimodal "
        f"--speculative-algorithm EAGLE --speculative-num-steps 3 "
        f"--speculative-eagle-topk 1 --speculative-num-draft-tokens 4 "
        f"--enable-multi-layer-eagle --trust-remote-code "
        f"--host 0.0.0.0 --port {port} {extra}".strip()
    )


def llama_cpp_notes() -> dict[str, Any]:
    return {
        "fork": "https://github.com/stepfun-ai/llama.cpp.git",
        "branch": "step3.7",
        "gguf_quant_sizes_gb": {
            "Q4_K_S": 111.5,
            "IQ4_XS": 104.99,
            "Q3_K_L": 102.5,
            "mm_projector_fp16": 3.97,
        },
        "runtime_overhead_gb": 7,
        "min_unified_memory_gb": 120,
        "recommended_unified_memory_gb": 128,
        "example_cli": "./llama-cli -m Step3.7_Q4_K_S.gguf -b 2048 -ub 2048 -fa on --temp 1.0 -p \"What's your name?\"",
    }


def deployment_card(backend: str = "vllm") -> dict[str, Any]:
    backend = backend.lower()
    if backend not in INFERENCE_BACKENDS:
        raise ValueError(f"backend must be one of {INFERENCE_BACKENDS}, got {backend!r}")

    card: dict[str, Any] = {
        "backend": backend,
        "hf_weights": HF_WEIGHTS,
        "docker_images": {
            "vllm": "vllm/vllm-openai:stepfun37",
            "sglang": "lmsysorg/sglang:dev-step-3.7-flash",
        },
        "parsers": {"reasoning": "step3p5", "tool_call": "step3p5"},
        "speculative": {"vllm": "mtp x3", "sglang": "EAGLE multi-layer"},
    }
    if backend == "vllm":
        card["commands"] = {
            "fp8": vllm_serve_command(precision="fp8"),
            "bf16": vllm_serve_command(precision="bf16"),
            "nvfp4": vllm_serve_command(precision="nvfp4"),
        }
    elif backend == "sglang":
        card["commands"] = {
            "bf16": sglang_serve_command(precision="bf16"),
            "fp8": sglang_serve_command(precision="fp8"),
            "nvfp4": sglang_serve_command(precision="nvfp4", tensor_parallel=4),
        }
    elif backend == "llama_cpp":
        card["llama_cpp"] = llama_cpp_notes()
    elif backend == "transformers":
        card["note"] = "Requires transformers>=5.0; use for debug/verification only."
        card["hf_id"] = HF_WEIGHTS["bf16"]
    return card
