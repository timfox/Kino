"""LLaVA-OneVision-2 framework card and reference experiment tables."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ltx_trainer.llava_ov2.codec_stream import (
    adaptive_gop_boundaries,
    bit_cost_per_bin,
    block_scores_from_saliency,
    motion_residual_saliency,
)
from ltx_trainer.llava_ov2.config import LLaVAOV2Config
from ltx_trainer.llava_ov2.grouper import group_visible_mask, image_single_group


def framework_card(cfg: LLaVAOV2Config | None = None) -> dict[str, Any]:
    cfg = cfg or LLaVAOV2Config()
    return {
        "name": "LLaVA-OneVision-2",
        "paper": "arXiv:2605.25979",
        "title": "Towards Next-Generation Perceptual Intelligence",
        "encoder": cfg.backbone_encoder,
        "llm": cfg.llm,
        "codec_stream": "bit-cost adaptive GOPs + motion-residual 2×2 block canvases",
        "training": "4-stage progressive recipe; codec-stream in Stage 4 for long video",
        "benchmark": f"JumpScore ({cfg.jumpscore_clips} clips)",
        "code": "https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2",
        "model": "https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
    }


def table_video_benchmarks() -> dict[str, dict[str, float]]:
    """Table 1 — 18-task video suite (+ JumpScore)."""
    return {
        "llava_ov2": {
            "videomme": 71.9,
            "mvbench": 66.2,
            "tempcompass": 74.5,
            "jumpscore": 74.9,
            "charades": 53.5,
            "activitynet": 53.8,
            "qvhighlights": 66.4,
            "vsi_bench": 70.9,
            "video_avg_18": 62.5,
        },
        "qwen3_vl": {
            "videomme": 71.4,
            "mvbench": 69.0,
            "tempcompass": 74.3,
            "jumpscore": 30.1,
            "charades": 48.3,
            "activitynet": 46.8,
            "qvhighlights": 59.4,
            "vsi_bench": 59.1,
            "video_avg_18": 58.2,
        },
    }


def table_spatial_benchmarks() -> dict[str, dict[str, float]]:
    """Table 2 — spatial reasoning subset."""
    return {
        "llava_ov2": {
            "crosspoint": 61.9,
            "tracespatial_3d": 31.0,
            "cv_bench_3d": 92.8,
            "spatial_avg_11": 63.5,
        },
        "qwen3_vl": {
            "crosspoint": 26.9,
            "tracespatial_3d": 8.0,
            "cv_bench_3d": 92.3,
            "spatial_avg_11": 58.2,
        },
    }


def table_tracking_jf() -> dict[str, dict[str, float]]:
    """Table 1 tracking + Table 6 J&F."""
    return {
        "llava_ov2": {"davis": 58.7, "mevis": 45.7, "revos_ref": 58.2, "revos_reason": 29.2, "avg_4": 48.0},
        "qwen3_vl": {"davis": 41.3, "mevis": 28.4, "revos_ref": 37.8, "revos_reason": 21.9, "avg_4": 32.4},
    }


def table_codec_vs_uniform() -> dict[str, float]:
    """Sec. 8 — matched-budget codec-stream vs frame sampling."""
    return {
        "jumpscore_codec_gain": 9.7,
        "temporal_grounding_avg_gain": 9.7,
        "jumpscore_128f_uniform": 45.4,
        "jumpscore_128f_codec": 74.9,
        "videomme_long_sub_codec_gain": 1.2,
    }


def table_training_stages() -> dict[str, dict[str, Any]]:
    """Sec. 4 — four-stage frame budgets."""
    return {
        "stage1": {"frames": 30, "codec": False, "video_caption_m": 4.2},
        "stage2": {"frames": 90, "codec": False, "instruct_m": 46.0},
        "stage3": {"frames": 384, "codec": False},
        "stage4": {"frames": 768, "codec": True, "spatial_m": 4.0},
    }


def table_codec_frame_budget_sweep() -> dict[str, dict[str, float]]:
    """Table 4 / Figure 7 — JumpScore at 128 frames."""
    return {
        "128_uniform": 45.4,
        "128_codec": 74.9,
    }


def training_step_demo(
    cfg: LLaVAOV2Config | None = None,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Smoke: GOP partition + saliency blocks + group-visible mask."""
    cfg = cfg or LLaVAOV2Config()
    dev = torch.device(device)

    p_bytes = torch.rand(448, device=dev) * 1000
    bins = 32
    costs = bit_cost_per_bin(p_bytes, bins)
    groups = adaptive_gop_boundaries(
        costs,
        k_tar=cfg.target_gop_count,
        l_min=max(1, int(cfg.gop_min_s / cfg.bin_duration_s)),
        l_max=max(1, int(cfg.gop_max_s / cfg.bin_duration_s)),
    )

    sal = motion_residual_saliency(
        torch.rand(256, 256, device=dev),
        torch.randint(100, 156, (256, 256), device=dev).float(),
    )
    blocks = block_scores_from_saliency(sal)

    n_tok = 64
    gids = torch.randint(0, len(groups), (n_tok,), device=dev)
    mask = group_visible_mask(gids)
    img_g = image_single_group(n_tok)

    stub = nn.Linear(cfg.hidden_dim, cfg.hidden_dim).to(dev)
    x = stub(torch.randn(1, n_tok, cfg.hidden_dim, device=dev))

    return {
        "num_gops": float(len(groups)),
        "block_map_elements": float(blocks.numel()),
        "group_mask_density": float(mask.mean()),
        "image_group_max": float(img_g.max()),
        "output_norm": float(x.detach().norm()),
    }


def evaluation_demo(*, device: str = "cpu") -> dict[str, Any]:
    step = training_step_demo(device=device)
    vid = table_video_benchmarks()
    spa = table_spatial_benchmarks()
    trk = table_tracking_jf()
    codec = table_codec_vs_uniform()
    return {
        **step,
        "jumpscore_ov2": vid["llava_ov2"]["jumpscore"],
        "jumpscore_qwen3": vid["qwen3_vl"]["jumpscore"],
        "jumpscore_gain": vid["llava_ov2"]["jumpscore"] - vid["qwen3_vl"]["jumpscore"],
        "video_avg_gain": vid["llava_ov2"]["video_avg_18"] - vid["qwen3_vl"]["video_avg_18"],
        "spatial_avg_gain": spa["llava_ov2"]["spatial_avg_11"] - spa["qwen3_vl"]["spatial_avg_11"],
        "tracking_jf_gain": trk["llava_ov2"]["avg_4"] - trk["qwen3_vl"]["avg_4"],
        "codec_jumpscore_gain_matched": codec["jumpscore_codec_gain"],
    }
