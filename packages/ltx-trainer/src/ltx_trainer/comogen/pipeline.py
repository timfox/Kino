"""CoMoGen training-free mask-guided I2V glue (Sec. 4, Fig. 2)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.comogen.attention import demo_attention_maps, identify_motion_layers, layer_attention_scores
from ltx_trainer.comogen.config import BASELINES, DATASETS, CoMoGenConfig
from ltx_trainer.comogen.mask_adapter import MaskAdapter, mask_to_delta
from ltx_trainer.comogen.metrics import evaluate_video
from ltx_trainer.comogen.schedule import cosine_schedule, inject_latent_residual


def dataset_card(cfg: CoMoGenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CoMoGenConfig()
    return {
        "name": "CoMoGen",
        "paper": "arXiv:2605.22996",
        "paradigm": "mask-guided controllable I2V with MaskAdapter + Motion Layers LoRA",
        "datasets": list(DATASETS),
        "baselines": list(BASELINES),
        "num_motion_layers": cfg.num_motion_layers,
        "extra_params_inference_pct": cfg.inference_param_pct,
        "extra_params_training_pct": cfg.training_param_pct,
        "backbone": "HunyuanVideo-I2V (MMDiT)",
    }


def benchmark_table_behave() -> dict[str, dict[str, float]]:
    """Table 3 BEHAVE column."""
    return {
        "goflow": {"ssim": 0.5044, "psnr": 19.21, "lpips": 0.1823, "fvd": 617.24},
        "interdyn": {"ssim": 0.5006, "psnr": 16.98, "lpips": 0.1893, "fvd": 720.14},
        "magicmotion": {"ssim": 0.7718, "psnr": 23.28, "lpips": 0.1191, "fvd": 694.41},
        "comogen": {"ssim": 0.7940, "psnr": 22.99, "lpips": 0.0721, "fvd": 327.63},
    }


def benchmark_table_clevrer() -> dict[str, dict[str, float]]:
    """Table 3 CLEVRER column."""
    return {
        "goflow": {"ssim": 0.9117, "psnr": 26.56, "lpips": 0.1085, "fvd": 341.01},
        "interdyn": {"ssim": 0.8275, "psnr": 25.16, "lpips": 0.1632, "fvd": 358.02},
        "magicmotion": {"ssim": 0.7612, "psnr": 23.16, "lpips": 0.3315, "fvd": 466.51},
        "comogen": {"ssim": 0.9252, "psnr": 27.12, "lpips": 0.1732, "fvd": 258.52},
    }


def ablation_table_behave() -> dict[str, dict[str, float]]:
    """Table 5 BEHAVE ablations."""
    return {
        "lora_non_motion_layers": {"ssim": 0.7787, "psnr": 21.81, "lpips": 0.0922, "fvd": 371.96},
        "no_cosine_schedule": {"ssim": 0.7938, "psnr": 22.99, "lpips": 0.0821, "fvd": 336.63},
        "comogen": {"ssim": 0.7940, "psnr": 22.99, "lpips": 0.0721, "fvd": 327.63},
    }


def layer_skip_benchmark() -> dict[str, dict[str, float]]:
    """Tables 1–2 layer-skipping analysis (reference)."""
    return {
        "tracking": {
            "skip_motion_layers": {"j": 60.1, "f": 59.9, "j_and_f": 60.0, "hota": 57.3},
            "skip_non_motion_layers": {"j": 77.7, "f": 81.1, "j_and_f": 79.4, "hota": 82.3},
        },
        "vqa": {
            "generated": 0.4540,
            "skip_motion_layers": 0.2742,
            "skip_non_motion_layers": 0.4068,
        },
    }


def sparsify_mask_sequence(mask: Tensor, every_n: int) -> Tensor:
    """Appendix D: retain one mask every N frames; skipped frames zeroed."""
    if every_n <= 1:
        return mask.clone()
    out = torch.zeros_like(mask)
    t = mask.shape[0]
    for i in range(0, t, every_n):
        out[i] = mask[i]
        if every_n >= 8 and i + every_n < t:
            # propagate subsequent mask to preceding skipped frame
            out[i + every_n - 1] = mask[min(i + every_n - 1, t - 1)]
    return out


def motion_layer_analysis(cfg: CoMoGenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CoMoGenConfig()
    mask, attn = demo_attention_maps(num_layers=cfg.num_dit_layers)
    scores = layer_attention_scores(mask, attn)
    motion, non_motion = identify_motion_layers(scores, cfg.num_motion_layers)
    return {
        "scores": scores,
        "motion_layers": motion,
        "non_motion_layers": non_motion[:5],  # truncate for readability
        "top_score": max(scores[i] for i in motion) if motion else 0.0,
        "bottom_score": min(scores[i] for i in motion) if motion else 0.0,
    }


def demo_mask_sequence(
    *,
    frames: int = 17,
    height: int = 64,
    width: int = 64,
) -> Tensor:
    """Binary mask sequence with moving subject blob."""
    mask = torch.zeros(frames, height, width)
    for t in range(frames):
        cx = 16 + t * 2
        cy = height // 2
        mask[t, cy - 6 : cy + 6, cx - 6 : cx + 6] = 1.0
    return mask


def comogen_denoise_step(
    z_t: Tensor,
    mask: Tensor,
    step: int,
    total_steps: int,
    adapter: MaskAdapter | None = None,
    cfg: CoMoGenConfig | None = None,
) -> dict[str, Tensor | float]:
    """One flow-matching step with cosine-weighted ΔZ injection."""
    cfg = cfg or CoMoGenConfig()
    delta = mask_to_delta(mask, adapter, cfg)
    if delta.shape != z_t.shape:
        delta = torch.nn.functional.interpolate(
            delta,
            size=z_t.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
    z_out = inject_latent_residual(z_t, delta, step, total_steps)
    w = cosine_schedule(total_steps)[step]
    return {"z_out": z_out, "delta_z": delta, "weight": w}


def propagate_demo(
    mask: Tensor,
    *,
    steps: int = 8,
    channels: int = 16,
    cfg: CoMoGenConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Toy latent propagation with mask residual injection."""
    cfg = cfg or CoMoGenConfig()
    adapter = MaskAdapter(cfg)
    t_lat = (mask.shape[0] + cfg.temporal_downsample - 1) // cfg.temporal_downsample
    h8, w8 = mask.shape[1] // cfg.spatial_downsample, mask.shape[2] // cfg.spatial_downsample
    z = torch.randn(1, cfg.latent_channels, t_lat, h8, w8)
    trajectory = [z.clone()]
    for s in range(steps):
        out = comogen_denoise_step(z, mask, s, steps, adapter, cfg)
        z = out["z_out"] * 0.95 + 0.05 * torch.randn_like(z)
        trajectory.append(z.clone())
    return z, torch.stack(trajectory)


def evaluate_comogen(
    pred_video: Tensor,
    target_video: Tensor,
    mask_seq: Tensor,
    *,
    cfg: CoMoGenConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or CoMoGenConfig()
    metrics = evaluate_video(pred_video, target_video)
    metrics["reference"] = {
        "ssim": cfg.reference_ssim,
        "psnr": cfg.reference_psnr,
        "lpips": cfg.reference_lpips,
        "fvd": cfg.reference_fvd,
    }
    return metrics


class MotionLayerLoRA(nn.Module):
    """Minimal LoRA stub for motion-layer attention projections (Sec. 4.2)."""

    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        self.lora_a = nn.Linear(dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_b(self.lora_a(x))
