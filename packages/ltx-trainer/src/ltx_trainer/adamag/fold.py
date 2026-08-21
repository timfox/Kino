"""Fold AdaMaG guidance hints into video latent shards (sampling metadata)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.adamag.config import AdamagConfig
from ltx_trainer.adamag.guidance import adamag_velocity, cfg_velocity, omega_schedule


def adamag_meta_block() -> dict[str, Any]:
    return {
        "adamag": {
            "arxiv_id": "2605.20079",
            "fold_role": "video_sampling_hint",
            "note": "Toy 1-D velocity probe stored for HQ / validation schedulers",
        }
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = AdamagConfig()
    omega_ref = cfg.sd3_high_guidance_omega
    beta = cfg.beta_default
    gamma = cfg.gamma_default
    omega_min = cfg.omega_min_default
    latents = data.get("latents")
    out = dict(data)
    out.update(adamag_meta_block())

    # 1-D probe from latent mean (does not alter stored latents)
    if latents is not None:
        try:
            import torch

            x = latents.float().mean().reshape(-1)[:8]
            x_list = x.detach().cpu().tolist()
        except Exception:
            x_list = [0.0, 0.1, 0.2, 0.1]
    else:
        x_list = [0.0, 0.1, 0.2, 0.1]

    v_u = [0.0] * len(x_list)
    v_c = [0.05 * (i + 1) for i in range(len(x_list))]
    t = 0.35
    v_cfg = cfg_velocity(v_u, v_c, omega_ref)
    v_adamag = adamag_velocity(
        v_u, v_c, x_list, t=t, omega_ref=omega_ref, beta=beta, gamma=gamma, omega_min=omega_min
    )
    omega_t = omega_schedule(t, omega_ref, omega_min=omega_min, gamma=gamma)

    out["adamag"].update(
        {
            "omega_t": round(omega_t, 4),
            "beta": beta,
            "omega_ref": omega_ref,
            "cfg_norm": round(sum(v * v for v in v_cfg) ** 0.5, 4),
            "adamag_norm": round(sum(v * v for v in v_adamag) ** 0.5, 4),
            "adamag_beats_cfg": sum(v * v for v in v_adamag) <= sum(v * v for v in v_cfg) * 1.5,
        }
    )
    return out
