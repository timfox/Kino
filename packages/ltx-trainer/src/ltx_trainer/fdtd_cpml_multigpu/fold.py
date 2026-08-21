"""AV-fold sidecar: wave/FDTD simulation hints on video latents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fdtd_cpml_multigpu.communication import compare_exchange
from ltx_trainer.fdtd_cpml_multigpu.constants import PAPER_ARXIV
from ltx_trainer.fdtd_cpml_multigpu.decomposition import select_best_decomposition


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("seismic", "wave", "acoustic", "fdtd", "electromagnetic")):
        domain = "acoustic_fdtd_cpml"
    else:
        domain = "generic_stencil"
    out["fdtd_cpml_multigpu"] = {
        "arxiv_id": PAPER_ARXIV,
        "domain_hint": domain,
        "decomposition": select_best_decomposition(800)["selected"],
        "exchange_hint": "peer" if "gpu" in caption or "cuda" in caption else "compare",
        "peer_speedup_anchor": compare_exchange(800)["speedup"],
    }
    return out
