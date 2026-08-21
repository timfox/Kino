"""AV-fold sidecar: Ozaki / fp8 emulation hints on video latents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fp8_ozaki.constants import PAPER_ARXIV, TABLE3_SPEEDUP_ANCHORS
from ltx_trainer.fp8_ozaki.tme import ozaki_speedup


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    workload = "dense_gemm" if any(w in caption for w in ("gemm", "matmul", "dense")) else "stencil_7pt"
    gpu = "B300" if "b300" in caption else "B200"
    out["fp8_ozaki"] = {
        "arxiv_id": PAPER_ARXIV,
        "workload_hint": workload,
        "gpu_hint": gpu,
        "moduli_count": 10,
        "ozaki_speedup_anchor": TABLE3_SPEEDUP_ANCHORS.get(workload, {}).get(gpu, 1.0),
        "tme_speedup": round(ozaki_speedup(0.5, gpu, moduli_count=10), 2),
    }
    return out
