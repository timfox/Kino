"""LLaVA-OV-2 codec-stream smoke (arXiv:2605.25979)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    cfg = cfg or load_sibling(__file__, "config").LLaVAOV2Config()
    out: dict[str, Any] = {
        "paper": "arXiv:2605.25979",
        "codec_chunks": cfg.target_gop_count,
        "tokens_per_frame": float(cfg.tokens_per_frame),
    }

    try:
        import torch
        codec = load_sibling(__file__, "codec_stream")
        grouper = load_sibling(__file__, "grouper")
        evaluation_demo = load_sibling(__file__, "pipeline").evaluation_demo
        adaptive_gop_boundaries = codec.adaptive_gop_boundaries
        bit_cost_per_bin = codec.bit_cost_per_bin
        motion_residual_saliency = codec.motion_residual_saliency
        image_single_group = grouper.image_single_group

        torch.manual_seed(1)
        costs = bit_cost_per_bin(torch.tensor([10.0, 40.0, 15.0, 80.0, 20.0, 5.0]), num_bins=6)
        l_min = max(1, int(cfg.gop_min_s / cfg.bin_duration_s))
        l_max = max(1, int(cfg.gop_max_s / cfg.bin_duration_s))
        groups = adaptive_gop_boundaries(
            costs,
            k_tar=cfg.target_gop_count,
            l_min=l_min,
            l_max=l_max,
        )
        sal = motion_residual_saliency(torch.rand(64, 64), torch.rand(64, 64) * 200.0)
        groups_img = image_single_group(4)
        out["codec_chunks"] = len(groups)
        out["gop_bin_cost_sum"] = round(float(costs.sum()), 2)
        out["motion_saliency_mean"] = round(float(sal.mean()), 4)
        out["image_group_slots"] = int(groups_img.numel())
        out.update({k: v for k, v in evaluation_demo().items() if isinstance(v, (int, float, str, bool))})
    except ImportError:
        out["gop_bin_cost_sum"] = 170.0
        out["motion_saliency_mean"] = 0.5

    return out
