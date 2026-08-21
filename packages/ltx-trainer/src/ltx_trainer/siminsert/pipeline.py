"""SimInsert dual-path insertion glue (Sec. III, Fig. 2)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.siminsert.attention import apply_value_guidance
from ltx_trainer.siminsert.config import BASELINES, PAPER_METRICS, SimInsertConfig
from ltx_trainer.siminsert.latent import latent_refresh, reconstruction_latent
from ltx_trainer.siminsert.metrics import evaluate_background_metrics


def demo_video_latents(
    *,
    frames: int = 8,
    tokens_per_frame: int = 16,
    dim: int = 32,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Synthetic original video x0, edited first frame, mask, and shared noise."""
    device = device or torch.device("cpu")
    x0 = torch.randn(frames, tokens_per_frame, dim, device=device)
    x0_edited = x0.clone()
    x0_edited[0] = x0_edited[0] + 0.5  # edited first frame
    mask = torch.zeros(frames, tokens_per_frame, device=device)
    mask[0, : tokens_per_frame // 4] = 1.0  # edited region on frame 0
    eps = torch.randn_like(x0)
    return x0, x0_edited, mask, eps


def siminsert_denoise_step(
    x_edited: Tensor,
    x0: Tensor,
    mask: Tensor,
    v_vis: Tensor,
    v_rec_vis: Tensor,
    *,
    t: float = 0.5,
    epsilon: Tensor | None = None,
    cfg: SimInsertConfig | None = None,
) -> dict[str, Tensor]:
    """One dual-path step: ReAC + sparse fusion + latent refresh."""
    cfg = cfg or SimInsertConfig()
    eps = epsilon if epsilon is not None else torch.randn_like(x0)

    x_rec = reconstruction_latent(x0, t, eps)
    v_star = apply_value_guidance(
        v_vis,
        v_rec_vis,
        mask,
        use_clone=cfg.use_regional_clone and not cfg.use_sparse_fusion,
        use_fusion=cfg.use_sparse_fusion,
        retention_p=cfg.fusion_retention_p,
    )
    if cfg.use_latent_refresh:
        x_star = latent_refresh(x_edited, x0, mask, t, eps)
    else:
        x_star = x_edited

    return {
        "x_rec": x_rec,
        "x_star": x_star,
        "v_star": v_star,
        "epsilon": eps,
    }


def propagate_demo(
    x0: Tensor,
    x0_edited_first: Tensor,
    mask: Tensor,
    *,
    steps: int = 4,
    cfg: SimInsertConfig | None = None,
) -> Tensor:
    """Toy temporal propagation with guidance at each step."""
    cfg = cfg or SimInsertConfig()
    x = x0.clone()
    x[0] = x0_edited_first[0]
    eps = torch.randn_like(x0)
    for i in range(steps):
        t = 1.0 - (i + 1) / (steps + 1)
        v_vis = x + 0.1 * torch.randn_like(x)
        v_rec = reconstruction_latent(x0, t, eps)
        out = siminsert_denoise_step(
            x,
            x0,
            mask,
            v_vis,
            v_rec,
            t=t,
            epsilon=eps,
            cfg=cfg,
        )
        x = out["x_star"]
    return x


def evaluate_insertion(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    prompt: str = "A rhino walks in the park.",
    cfg: SimInsertConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or SimInsertConfig()
    metrics = evaluate_background_metrics(pred, target, mask, prompt=prompt)
    metrics["reference"] = {
        "psnr": cfg.reference_psnr,
        "ssim": cfg.reference_ssim,
        "lpips": cfg.reference_lpips,
        "vfid": cfg.reference_vfid,
        "clip_i": cfg.reference_clip_i,
        "clip_t": cfg.reference_clip_t,
    }
    return metrics


def benchmark_table() -> dict[str, dict[str, float]]:
    """Paper Table I reference scores."""
    return {
        "pix2video": {"psnr": 28.96, "ssim": 0.7219, "lpips": 0.3852, "vfid": 1479.67, "clip_i": 0.9803, "clip_t": 0.2789},
        "fatezero": {"psnr": 29.18, "ssim": 0.6345, "lpips": 0.3835, "clip_i": 0.9737, "clip_t": 0.2499},
        "consisti2v": {"psnr": 30.52, "ssim": 0.5903, "lpips": 0.4721, "clip_i": 0.9788, "clip_t": 0.2837},
        "anyv2v": {"psnr": 29.75, "ssim": 0.6815, "lpips": 0.2630, "vfid": 1240.54, "clip_i": 0.9814, "clip_t": 0.2867},
        "siminsert": {"psnr": 36.26, "ssim": 0.8671, "lpips": 0.1471, "vfid": 1062.92, "clip_i": 0.9923, "clip_t": 0.2825},
    }


def dataset_card(cfg: SimInsertConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SimInsertConfig()
    return {
        "name": "SimInsert",
        "paper": "arXiv:2605.23245",
        "paradigm": "training-free I2V object insertion",
        "num_frames_eval": cfg.num_frames,
        "resolution": f"{cfg.height}p",
        "mechanisms": ["regional_attention_clone", "sparse_attention_fusion", "latent_refresh"],
        "baselines": list(BASELINES),
        "metrics": list(PAPER_METRICS),
        "backbones": ["Wan2.1", "CogVideoX-I2V", "LTX-Video"],
    }
