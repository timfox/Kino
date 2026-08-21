"""AV-fold sidecar: HPC code-porting / CUDA translation hints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deopt_reopt.constants import PAPER_ARXIV


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("conv2d", "gemm", "fft", "stencil", "hpc", "fdtd")):
        regime = "hpc_kernel"
    elif any(w in caption for w in ("cuda", "gpu port", "deopt", "neon", "simd")):
        regime = "cpp_to_cuda_port"
    elif any(w in caption for w in ("llm", "code translation", "refactor")):
        regime = "llm_porting"
    else:
        regime = "generic_compute"
    out["deopt_reopt"] = {
        "arxiv_id": PAPER_ARXIV,
        "regime_hint": regime,
        "workflow_hint": "deopt_reopt" if "neon" in caption or "optimized" in caption else "direct",
        "model_hint": "Q235" if "qwen" in caption else "O120",
        "kernel_count": 12,
    }
    return out
