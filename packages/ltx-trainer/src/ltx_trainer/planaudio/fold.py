"""Fold PlanAudio compositional proxies into LTX audio_latents shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.planaudio.config import PlanAudioConfig
from ltx_trainer.planaudio.hooks import score_prompt_composition
from ltx_trainer.planaudio.scenarios import classify_free_form_prompt


def planaudio_meta_block() -> dict[str, Any]:
    return {
        "planaudio": {
            "arxiv_id": "2605.28063",
            "fold_role": "compositional_unified_audio_proxy",
            "task": "free_form_text_to_unified_audio",
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach scenario + compositional readiness proxies from latent statistics."""
    cfg = PlanAudioConfig()
    latents = data.get("latents")
    caption = str(data.get("caption") or data.get("prompt") or "")
    out = dict(data)
    out.update(planaudio_meta_block())

    scenario = classify_free_form_prompt(caption) if caption else "sound"
    comp = score_prompt_composition(caption) if caption else {"semantic_coverage_factor_proxy": 0.0}

    if latents is None:
        out["planaudio"].update({"scenario": scenario, **comp})
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size < 8:
        flat = np.pad(flat, (0, 8 - flat.size))
    t_axis = 0 if arr.ndim >= 2 and arr.shape[0] <= 64 else None
    if t_axis is not None and arr.shape[t_axis] > 1:
        temporal_var = float(np.mean(np.abs(np.diff(arr, axis=t_axis))))
    else:
        temporal_var = float(np.std(flat))
    energy = float(np.mean(np.abs(flat)))
    composite_complexity = float(np.clip(0.5 * temporal_var + 0.5 * min(energy, 1.0), 0.0, 1.0))

    out["planaudio"].update(
        {
            "scenario": scenario,
            "semantic_coverage_factor_proxy": comp.get("semantic_coverage_factor_proxy", 0.0),
            "composite_complexity_proxy": round(composite_complexity, 4),
            "latent_cot_steps_ref": cfg.latent_cot_steps,
            "unified_audio_ready": scenario == "composite" or composite_complexity > 0.35,
        }
    )
    return out
