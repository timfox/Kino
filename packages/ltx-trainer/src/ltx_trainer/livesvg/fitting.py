"""Progressive target-video fitting schedule (Sec. 3.4.3–3.4.4)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.livesvg.config import LiveSVGConfig
from ltx_trainer.livesvg.homography import compose_path_motion, homography_from_similarity, identity_homography
from ltx_trainer.livesvg.losses import livesvg_total_loss


def effective_progressive_interval(
    opt_steps: int,
    num_keyframes: int,
    default_interval: int,
) -> int:
    """Shrink interval for short smoke runs so multiple keyframes actually activate."""
    if num_keyframes <= 1:
        return max(1, opt_steps)
    cap = max(1, opt_steps // max(num_keyframes - 1, 1))
    return max(1, min(default_interval, cap))


def progressive_activation_schedule(
    total_iters: int,
    num_keyframes: int,
    *,
    interval: int,
) -> list[int]:
    """Return active keyframe count after each optimization segment."""
    if num_keyframes < 1:
        raise ValueError("num_keyframes must be positive")
    active = 1
    schedule = [active]
    for step in range(1, total_iters + 1):
        if step % interval == 0 and active < num_keyframes:
            active += 1
        schedule.append(active)
    return schedule[:total_iters]


def copy_keyframe_motion(
    homographies: Tensor,
    path_deltas: Tensor,
    src_k: int,
    dst_k: int,
) -> None:
    """In-place copy group homographies and path offsets between keyframes."""
    homographies[dst_k].copy_(homographies[src_k])
    path_deltas[dst_k].copy_(path_deltas[src_k])


def tracking_init_fallback(
    tracked_delta: Tensor,
    previous_delta: Tensor,
    *,
    tracked_mse: float,
    previous_mse: float,
) -> Tensor:
    """Conservative fallback when TAPNext init is worse than copy-prev (Sec. 3.4.4)."""
    return tracked_delta if tracked_mse <= previous_mse else previous_delta


def toy_synthetic_fitting_step(
    *,
    num_paths: int = 6,
    canvas: int = 32,
    cfg: LiveSVGConfig | None = None,
) -> dict[str, float]:
    """Differentiable smoke: optimize homography + offsets against a moving disk target."""
    cfg = cfg or LiveSVGConfig()
    torch.manual_seed(7)
    device = torch.device("cpu")
    canonical = torch.rand(num_paths, 2, device=device)
    center = canonical.mean(dim=0)
    adjacency = [(i, i + 1) for i in range(num_paths - 1)]
    g1_triplets = [(max(0, i - 1), i, min(num_paths - 1, i + 1)) for i in range(1, num_paths - 1)]
    g1_enforce = [True] * len(g1_triplets)

    homographies = identity_homography(device=device).unsqueeze(0).repeat(1, 1, 1).requires_grad_(True)
    deltas = torch.zeros(1, num_paths, 2, device=device, requires_grad=True)

    target = torch.zeros(1, 1, canvas, canvas, device=device)
    target[0, 0, canvas // 4 : 3 * canvas // 4, canvas // 4 : 3 * canvas // 4] = 1.0

    opt = torch.optim.Adam(
        [
            {"params": [homographies], "lr": cfg.lr_homography},
            {"params": [deltas], "lr": cfg.lr_path_offsets},
        ],
        betas=(0.9, 0.9),
        eps=1e-6,
    )

    last: dict[str, Tensor] = {}
    for _ in range(12):
        moved = compose_path_motion(
            canonical,
            center=center,
            local_delta=deltas[0],
            homography=homographies[0],
        )
        rendered = torch.zeros_like(target)
        for p in moved:
            xi = int(p[0].clamp(0, canvas - 1).item())
            yi = int(p[1].clamp(0, canvas - 1).item())
            rendered[0, 0, yi, xi] = 1.0
        last = livesvg_total_loss(
            rendered,
            target,
            deltas=deltas[0],
            adjacency=adjacency,
            canonical_xy=canonical,
            foreground_mask=target[0, 0],
            control_points=moved,
            g1_triplets=g1_triplets,
            cfg_weights=(cfg.lambda_mse, cfg.lambda_spatial, cfg.lambda_g1, cfg.lambda_sdf),
            spatial_sigma=cfg.spatial_sigma_frac * canvas,
            g1_enforce=g1_enforce,
            sdf_margin=cfg.sdf_margin_px,
            blur_kernel=cfg.gaussian_kernel_size,
            blur_sigma=cfg.gaussian_sigma,
        )
        opt.zero_grad()
        last["total"].backward()
        opt.step()

    return {k: round(float(v.detach()), 4) for k, v in last.items()}


def pipeline_stage_summary() -> dict[str, Any]:
    return {
        "stage1": ["semantic_grouping", "sphere_packing_recolor", "layer_order", "target_i2v", "gemini_filter"],
        "stage2": ["diffvg_fit", "progressive_keyframes", "tapnext_init"],
        "decoupled": True,
    }
