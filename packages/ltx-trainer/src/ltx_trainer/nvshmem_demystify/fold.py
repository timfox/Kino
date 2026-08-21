"""AV-fold sidecar: GPU cluster / expert-parallel communication hints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvshmem_demystify.constants import PAPER_ARXIV, NVSHMEM_VERSION


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("deepep", "moe", "expert parallel", "mixture of experts")):
        regime = "deepep_moe"
    elif any(w in caption for w in ("nvshmem", "ibgda", "symmetric heap", "pgas")):
        regime = "nvshmem_pgas"
    elif any(w in caption for w in ("nccl", "allreduce", "collective")):
        regime = "nccl_collective"
    elif any(w in caption for w in ("hpc", "multi-gpu", "infiniBand", "nvlink")):
        regime = "gpu_cluster_comm"
    else:
        regime = "distributed_gpu"
    out["nvshmem_demystify"] = {
        "arxiv_id": PAPER_ARXIV,
        "regime_hint": regime,
        "nvshmem_version": NVSHMEM_VERSION,
        "transport_hint": "ibgda" if "inter" in caption or "node" in caption else "p2p_nvlink",
        "complement_nccl": True,
    }
    return out
