"""AV-fold sidecar: Dyna-Pruner co-pruning metadata on latent saves."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dyna_pruner.config import DynaPrunerConfig
from ltx_trainer.dyna_pruner.env import dyna_pruner_enabled
from ltx_trainer.dyna_pruner.inference import metadata_from_latent


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    cfg = DynaPrunerConfig()
    existing = out.get("dyna_pruner")
    if isinstance(existing, dict) and existing.get("skip_blocks"):
        return out
    latents = data.get("latents")
    meta: dict[str, Any] = {"arxiv_id": cfg.paper_arxiv, "enabled": dyna_pruner_enabled()}
    if latents is not None:
        try:
            import torch

            t = latents if isinstance(latents, torch.Tensor) else torch.as_tensor(latents)
            meta.update(metadata_from_latent(t))
        except Exception:
            meta["regime_hint"] = "spatiotemporal_media"
    else:
        caption = str(
            (data.get("meta") or {}).get("caption", "") if isinstance(data.get("meta"), dict) else data.get("caption", "")
        ).lower()
        if any(w in caption for w in ("weather", "radar", "storm", "traffic", "ocean", "sky")):
            meta["regime_hint"] = "high_dynamic_range_spatiotemporal"
        else:
            meta["regime_hint"] = "generic_video"
    out["dyna_pruner"] = meta
    return out
