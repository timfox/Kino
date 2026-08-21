"""Helix4D framework card and reference experiment tables."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.helix4d.attention import (
    apply_cross_frame_attention,
    mask_pattern_name,
    sliding_window_anchor_mask,
)
from ltx_trainer.helix4d.conditioning import (
    build_frame_timesteps,
    flow_matching_loss_mask,
    inject_anchor_latent,
)
from ltx_trainer.helix4d.config import Helix4DConfig
from ltx_trainer.helix4d.rope import apply_rope, rope_4d, split_spatial_rope, spatial_rope_3d


def framework_card(cfg: Helix4DConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Helix4DConfig()
    return {
        "name": "Helix4D",
        "paper": "arXiv:2605.26109",
        "title": "Complex 4D Mesh Generation",
        "backbone": "Trellis2 (O-Voxels) extended to video-conditioned 4D",
        "stages": list(cfg.stages),
        "design": {
            "attention": f"sliding window w={cfg.attention_window} + first-frame anchor",
            "rope": f"4D ReRoPE α={cfg.rope_alpha} (repurpose low-frequency spatial bands)",
            "conditioning": "frozen Trellis2 first frame as clean anchor",
        },
        "training": {
            "frames": cfg.num_frames,
            "iters": cfg.train_iters,
            "lr": cfg.learning_rate,
            "gpus": 32,
        },
        "project": "https://snap-research.github.io/helix4d/",
    }


def table_helix4d_bench() -> dict[str, dict[str, float | str]]:
    """Table 1 — Helix4DBench (52 videos)."""
    return {
        "ss4d": {
            "clip": 0.8468,
            "ulip2": 0.4191,
            "uni3d": 0.3739,
            "dreamsim": 0.2745,
            "fvd": 1046.0,
            "win_rate": 0.712,
        },
        "shapegen4d": {
            "clip_n": 0.6876,
            "ulip2": 0.4169,
            "uni3d": 0.3687,
            "win_rate": 0.679,
        },
        "mesh4d": {
            "clip": 0.8533,
            "clip_n": 0.7399,
            "ulip2": 0.4353,
            "uni3d": 0.3691,
            "dreamsim": 0.3025,
            "fvd": 979.0,
            "win_rate": 0.798,
        },
        "motion_3_to_4": {
            "clip": 0.8674,
            "clip_n": 0.7442,
            "ulip2": 0.4331,
            "uni3d": 0.3768,
            "dreamsim": 0.2697,
            "fvd": 955.0,
            "win_rate": 0.807,
        },
        "actionmesh": {
            "clip_n": 0.7156,
            "ulip2": 0.4226,
            "uni3d": 0.3761,
            "win_rate": 0.898,
        },
        "ours": {
            "clip": 0.8723,
            "clip_n": 0.7569,
            "ulip2": 0.4600,
            "uni3d": 0.4063,
            "dreamsim": 0.2593,
            "fvd": 950.0,
            "win_rate": "ref",
        },
    }


def table_actionbench_texverse() -> dict[str, dict[str, float]]:
    """Table 2 — CD-3D / CD-4D on ActionBench and TexVerse test."""
    return {
        "ss4d": {"cd_3d": 0.105, "cd_4d": 0.120, "cd_3d_tv": 0.052, "cd_4d_tv": 0.103},
        "shapegen4d": {"cd_3d": 0.056, "cd_4d": 0.170, "cd_3d_tv": 0.075, "cd_4d_tv": 0.129},
        "mesh4d": {"cd_3d": 0.096, "cd_4d": 0.141, "cd_3d_tv": 0.058, "cd_4d_tv": 0.100},
        "motion_3_to_4": {"cd_3d": 0.068, "cd_4d": 0.114, "cd_3d_tv": 0.056, "cd_4d_tv": 0.108},
        "actionmesh": {"cd_3d": 0.053, "cd_4d": 0.081, "cd_3d_tv": 0.056, "cd_4d_tv": 0.111},
        "ours": {"cd_3d": 0.051, "cd_4d": 0.093, "cd_3d_tv": 0.045, "cd_4d_tv": 0.097},
    }


def table_component_ablation() -> dict[str, dict[str, float]]:
    """Table 3 — component ablation (TexVerse held-out)."""
    return {
        "w/o_first_frame_cond": {"cd_3d": 0.0488, "ulip2": 0.2809, "uni3d": 0.2554},
        "w/o_4d_rotary": {"cd_3d": 0.0478, "ulip2": 0.2819, "uni3d": 0.2584},
        "w/o_our_attention": {"cd_3d": 0.0467, "ulip2": 0.2819, "uni3d": 0.2530},
        "ours": {"cd_3d": 0.0464, "ulip2": 0.2881, "uni3d": 0.2589},
    }


def table_attention_patterns() -> dict[str, dict[str, float]]:
    """Table 4 — cross-frame attention patterns (time normalized to ours=1)."""
    return {
        "full_attention": {"cd_3d": 0.0470, "cd_4d": 0.1009, "ulip2": 0.2835, "uni3d": 0.2447, "time": 2.3},
        "causal": {"cd_3d": 0.0491, "cd_4d": 0.1038, "ulip2": 0.2838, "uni3d": 0.2564, "time": 1.5},
        "sliding_window": {"cd_3d": 0.0468, "cd_4d": 0.1085, "ulip2": 0.2870, "uni3d": 0.2538, "time": 0.9},
        "spatial": {"cd_3d": 0.0471, "cd_4d": 0.1011, "ulip2": 0.2791, "uni3d": 0.2548, "time": 0.4},
        "ours": {"cd_3d": 0.0454, "cd_4d": 0.0970, "ulip2": 0.2879, "uni3d": 0.2576, "time": 1.0},
    }


def table_rope_ratio_ablation() -> dict[float, float]:
    """Table A1 — CD-3D vs full spatial RoPE (α ratio)."""
    return {0.0: 0.0181, 0.2: 0.0098, 0.4: 0.0079, 0.6: 0.0079, 0.8: 0.0079, 1.0: 0.0}


def evaluation_demo(*, device: str = "cpu") -> dict[str, Any]:
    """Smoke: anchor mask, 4D RoPE, first-frame conditioning."""
    cfg = Helix4DConfig()
    dev = torch.device(device)
    f, s_pf = 4, 6
    mask = sliding_window_anchor_mask(f, s_pf, window_half=cfg.attention_window // 2)
    feats = torch.randn(1, f * s_pf, cfg.head_dim, device=dev)
    out = apply_cross_frame_attention(feats, mask.to(dev))

    coords = torch.randint(0, cfg.voxel_grid, (s_pf, 3), device=dev)
    frames = torch.arange(f, device=dev)
    rot = rope_4d(coords, frames[0], cfg)
    x = torch.randn(cfg.head_dim, device=dev)
    x_rot = apply_rope(x, rot[0])

    latents = torch.randn(f, 8, device=dev)
    anchor = torch.randn(8, device=dev)
    latents_inj = inject_anchor_latent(latents, anchor)
    timesteps = build_frame_timesteps(f)
    loss_mask = flow_matching_loss_mask(f)

    spatial = spatial_rope_3d(coords.float(), cfg.head_dim, theta=cfg.rope_theta)
    _, ident_low = split_spatial_rope(spatial, cfg.rope_alpha)

    return {
        "mask_density": float(mask.mean()),
        "attention_out_shape": list(out.shape),
        "rope_rot_shape": list(rot.shape),
        "anchor_frame_tau": float(timesteps[0]),
        "loss_frames": int(loss_mask.sum()),
        "ident_low_shape": list(ident_low.shape),
        "ours_cd_3d_actionbench": table_actionbench_texverse()["ours"]["cd_3d"],
        "ours_ulip2_bench": table_helix4d_bench()["ours"]["ulip2"],
        "ours_pattern_cd_4d": table_attention_patterns()["ours"]["cd_4d"],
    }
