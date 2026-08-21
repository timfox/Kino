"""AV-fold sidecar: collective / MoE hints on video latents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pccl.constants import PAPER_ARXIV
from ltx_trainer.pccl.process_group import speedup_vs_direct


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("moe", "expert", "all-to-all", "alltoall")):
        collective = "all_to_all"
        n_groups = 2
    elif any(w in caption for w in ("tensor parallel", "all-gather", "allgather")):
        collective = "all_gather"
        n_groups = 1
    else:
        collective = "all_reduce"
        n_groups = 1
    out["pccl"] = {
        "arxiv_id": PAPER_ARXIV,
        "collective_hint": collective,
        "process_group_speedup_stub": speedup_vs_direct(n_process_groups=n_groups),
        "synthesizer": "pccl_ten_bfs",
    }
    return out
