"""LTX audio_latents sidecar: FoG-A / REPA readiness proxies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ag_repa.config import AgRepaConfig


def ag_repa_meta_block(cfg: AgRepaConfig | None = None) -> dict[str, Any]:
    c = cfg or AgRepaConfig()
    return {
        "ag_repa": {
            "arxiv_id": c.paper_arxiv,
            "fold_role": "causal_layer_repa_proxy",
            "top_k": c.top_k_layers,
            "paper_speech_top3": list(c.paper_top3_speech_fog),
        }
    }


def _latent_early_sensitivity(latents: Any) -> float:
    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size < 16:
        return 0.05
    early = flat[: flat.size // 4]
    late = flat[flat.size // 2 :]
    ratio = float(np.std(early) / (np.std(late) + 1e-9))
    return float(np.clip(ratio / 4.0, 0.02, 0.25))


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach FoG-A-style early-layer sensitivity proxy for FM audio latents."""
    c = AgRepaConfig()
    out = dict(data)
    out.update(ag_repa_meta_block(c))
    domain = str(data.get("domain") or data.get("modality") or "speech")
    fog_proxy = _latent_early_sensitivity(data.get("latents"))
    out["ag_repa"].update(
        {
            "domain": domain,
            "fog_a_proxy": round(fog_proxy, 4),
            "causal_bottleneck_ready": fog_proxy > 0.06,
            "repa_align_ready": bool(data.get("caption") or data.get("prompt")),
            "scd_note": "high_lasp_deep_vs_fog_early",
        }
    )
    return out
