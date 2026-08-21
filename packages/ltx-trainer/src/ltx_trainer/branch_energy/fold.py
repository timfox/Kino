"""AV-fold sidecar: power-quality / branch-energy localization proxies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.branch_energy.constants import PAPER_ARXIV


def branch_energy_meta_block() -> dict[str, Any]:
    return {
        "branch_energy": {
            "arxiv_id": PAPER_ARXIV,
            "fold_role": "branch_level_energy_localization",
            "method": "tellegen_branch_balance",
        }
    }


def _harmonic_spread_proxy(arr: np.ndarray) -> float:
    """Proxy for nonsinusoidal / switched-waveform content."""
    if arr.ndim < 4:
        return 0.5
    x = arr.astype(np.float64)
    diffs = np.diff(x, axis=0)
    return float(np.clip(np.std(diffs) / (np.std(x) + 1e-8), 0.0, 1.0))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(branch_energy_meta_block())

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("three-phase", "3-phase", "three phase", "wye", "delta")):
        regime = "three_phase_load"
    elif any(w in caption for w in ("triac", "switched resistive", "phase control")):
        regime = "switched_resistive"
    elif any(w in caption for w in ("open phase", "de leon", "cohen", "ghost current")):
        regime = "open_phase_paradox"
    elif any(w in caption for w in ("reactive power", "p-q", "ieee 1459", "cpc", "fbd")):
        regime = "power_theory"
    elif any(w in caption for w in ("branch energy", "tellegen", "joule", "localization")):
        regime = "branch_localization"
    else:
        regime = "unknown_power"

    latents = data.get("latents")
    if latents is None:
        out["branch_energy"].update({"regime_hint": regime, "has_latents": False})
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    harmonic = _harmonic_spread_proxy(arr)
    temporal = float(np.clip(1.0 / (1.0 + np.var(np.diff(arr.astype(np.float64), axis=0))), 0.0, 1.0))
    localization_readiness = float(np.clip(0.5 * harmonic + 0.5 * temporal, 0.0, 1.0))
    balance_proxy = float(np.clip(1.0 - abs(arr.mean()) / (arr.std() + 1e-8), 0.0, 1.0))

    out["branch_energy"].update(
        {
            "regime_hint": regime,
            "has_latents": True,
            "harmonic_spread_proxy": round(harmonic, 4),
            "localization_readiness_proxy": round(localization_readiness, 4),
            "tellegen_balance_proxy": round(balance_proxy, 4),
        }
    )
    return out
