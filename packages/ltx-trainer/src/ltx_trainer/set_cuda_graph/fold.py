"""AV-fold sidecar: CUDA graph / GPU pipeline scheduling hints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.set_cuda_graph.constants import PAPER_ARXIV


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("cuda graph", "set scheduling", "stream event", "work steal")):
        regime = "set_cuda_graph"
    elif any(w in caption for w in ("gpu pipeline", "kernel launch", "nsight", "memcpy")):
        regime = "gpu_pipeline"
    elif any(w in caption for w in ("gemm", "sobel", "hotspot", "sssp", "knn")):
        regime = "set_workload"
    else:
        regime = "gpu_compute"
    out["set_cuda_graph"] = {
        "arxiv_id": PAPER_ARXIV,
        "regime_hint": regime,
        "framework": "SET",
        "scheduling": "event_chained_work_stealing",
    }
    return out
