"""Multi-frame fitting experiments (synthetic target video)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.livesvg.config import LiveSVGConfig
from ltx_trainer.livesvg.fitting import effective_progressive_interval, progressive_activation_schedule
from ltx_trainer.livesvg.homography import compose_path_motion, homography_from_similarity
from ltx_trainer.livesvg.losses import livesvg_total_loss
from ltx_trainer.livesvg.paper_tables import table_aniclipart_quantitative
from ltx_trainer.livesvg.target_video import clean_target_background, synthetic_translating_disk


def _splat_points_to_frame(
    points: Tensor,
    *,
    canvas: int,
    device: torch.device,
) -> Tensor:
    frame = torch.zeros(1, 1, canvas, canvas, device=device)
    for p in points:
        xi = int(p[0].clamp(0, canvas - 1).item())
        yi = int(p[1].clamp(0, canvas - 1).item())
        frame[0, 0, yi, xi] = 1.0
    return frame


def _keyframe_total_loss(
    k: int,
    *,
    canonical: Tensor,
    center: Tensor,
    deltas: Tensor,
    homographies: Tensor,
    target: Tensor,
    canvas: int,
    device: torch.device,
    adjacency: list[tuple[int, int]],
    g1_triplets: list[tuple[int, int, int]],
    g1_enforce: list[bool],
    cfg: LiveSVGConfig,
) -> Tensor:
    moved = compose_path_motion(
        canonical,
        center=center,
        local_delta=deltas[k],
        homography=homographies[k],
    )
    rendered = _splat_points_to_frame(moved, canvas=canvas, device=device)
    return livesvg_total_loss(
        rendered,
        target[k : k + 1],
        deltas=deltas[k],
        adjacency=adjacency,
        canonical_xy=canonical,
        foreground_mask=target[k, 0],
        control_points=moved,
        g1_triplets=g1_triplets,
        cfg_weights=(cfg.lambda_mse, cfg.lambda_spatial, cfg.lambda_g1, cfg.lambda_sdf),
        spatial_sigma=cfg.spatial_sigma_frac * canvas,
        g1_enforce=g1_enforce,
        sdf_margin=cfg.sdf_margin_px,
        blur_kernel=cfg.gaussian_kernel_size,
        blur_sigma=cfg.gaussian_sigma,
    )["total"]


def run_synthetic_fitting_experiment(
    *,
    cfg: LiveSVGConfig | None = None,
    num_paths: int = 8,
    canvas: int = 48,
    opt_steps: int = 80,
    seed: int = 11,
    return_tensors: bool = False,
) -> dict[str, Any]:
    """Optimize group homography + path offsets against a translating-disk target."""
    cfg = cfg or LiveSVGConfig()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    disk_radius = max(4, canvas // 10)
    frame_dx = canvas / max(cfg.num_keyframes, 1)
    frame_dy = 0.5
    target = clean_target_background(
        synthetic_translating_disk(
            num_frames=cfg.num_keyframes,
            height=canvas,
            width=canvas,
            radius=disk_radius,
            dx=frame_dx,
            dy=frame_dy,
        ),
        threshold=0.5,
    )

    cx0 = canvas * 0.25
    cy0 = canvas * 0.5
    angles = torch.linspace(0, 2 * torch.pi, num_paths + 1, device=device)[:-1]
    canonical = torch.stack(
        [
            cx0 + disk_radius * 0.65 * torch.cos(angles),
            cy0 + disk_radius * 0.65 * torch.sin(angles),
        ],
        dim=1,
    )
    center = canonical.mean(dim=0)
    adjacency = [(i, i + 1) for i in range(num_paths - 1)]
    g1_triplets = [(max(0, i - 1), i, min(num_paths - 1, i + 1)) for i in range(1, num_paths - 1)]
    g1_enforce = [True] * len(g1_triplets)

    homographies = torch.stack(
        [
            homography_from_similarity(tx=frame_dx * k, ty=frame_dy * k, device=device)
            for k in range(cfg.num_keyframes)
        ],
        dim=0,
    ).requires_grad_(True)
    deltas = torch.zeros(cfg.num_keyframes, num_paths, 2, device=device, requires_grad=True)

    opt = torch.optim.Adam(
        [
            {"params": [homographies], "lr": cfg.lr_homography},
            {"params": [deltas], "lr": cfg.lr_path_offsets},
        ],
        betas=(0.9, 0.9),
        eps=1e-6,
    )

    interval = effective_progressive_interval(
        opt_steps, cfg.num_keyframes, cfg.progressive_interval_iters
    )
    schedule = progressive_activation_schedule(opt_steps, cfg.num_keyframes, interval=interval)
    history: list[float] = []
    ref_history: list[float] = []

    loss_kw = dict(
        canonical=canonical,
        center=center,
        deltas=deltas,
        homographies=homographies,
        target=target,
        canvas=canvas,
        device=device,
        adjacency=adjacency,
        g1_triplets=g1_triplets,
        g1_enforce=g1_enforce,
        cfg=cfg,
    )

    with torch.no_grad():
        initial_mse = float(
            torch.nn.functional.mse_loss(
                _splat_points_to_frame(
                    compose_path_motion(
                        canonical,
                        center=center,
                        local_delta=deltas[0],
                        homography=homographies[0],
                    ),
                    canvas=canvas,
                    device=device,
                ),
                target[0:1],
            ).item()
        )
        ref_history.append(float(_keyframe_total_loss(0, **loss_kw).item()))

    for step in range(opt_steps):
        active_k = schedule[min(step, len(schedule) - 1)]
        opt.zero_grad()
        losses_k = [_keyframe_total_loss(k, **loss_kw) for k in range(active_k)]
        total = torch.stack(losses_k).mean()
        total.backward()
        opt.step()
        history.append(float(total.detach()))
        ref_history.append(float(_keyframe_total_loss(0, **loss_kw).item()))

    with torch.no_grad():
        final_mse = []
        for k in range(cfg.num_keyframes):
            moved = compose_path_motion(canonical, center=center, local_delta=deltas[k], homography=homographies[k])
            rendered = _splat_points_to_frame(moved, canvas=canvas, device=device)
            final_mse.append(float(torch.nn.functional.mse_loss(rendered, target[k : k + 1]).item()))

    out: dict[str, Any] = {
        "num_keyframes": cfg.num_keyframes,
        "num_paths": num_paths,
        "opt_steps": opt_steps,
        "progressive_interval": interval,
        "loss_first": round(history[0], 4),
        "loss_last": round(history[-1], 4),
        "ref_loss_first": round(ref_history[0], 4),
        "ref_loss_last": round(ref_history[-1], 4),
        "loss_decreased": ref_history[-1] < ref_history[0],
        "initial_per_frame_mse": round(initial_mse, 4),
        "per_frame_mse_mean": round(sum(final_mse) / len(final_mse), 4),
        "mse_improved": (sum(final_mse) / len(final_mse)) < initial_mse,
        "active_keyframes_end": schedule[-1],
        "source": "synthetic",
    }
    if return_tensors:
        out["_homographies"] = homographies.detach()
        out["_path_deltas"] = deltas.detach()
    return out


def run_full_experiment(
    *,
    cfg: LiveSVGConfig | None = None,
    opt_steps: int = 120,
    seed: int = 11,
) -> dict[str, Any]:
    """Fitting experiment with paper Table 2 reference metrics attached."""
    cfg = cfg or LiveSVGConfig()
    exp = run_synthetic_fitting_experiment(cfg=cfg, opt_steps=opt_steps, seed=seed, return_tensors=True)
    ref = table_aniclipart_quantitative()["LiveSVG (Veo 3.1)"]
    exp["metrics"] = {k: float(v) for k, v in ref.items() if k in ("XCLIP", "LPIPS", "SSIM", "DOVER")}
    return exp


def export_experiment_bundle(path: str | Path, experiment: dict[str, Any]) -> Path:
    from ltx_trainer.livesvg.bundle import export_motion_bundle

    if "_homographies" not in experiment or "_path_deltas" not in experiment:
        raise KeyError("experiment missing tensors; run with return_tensors=True")
    return export_motion_bundle(
        path,
        homographies=experiment["_homographies"],
        path_deltas=experiment["_path_deltas"],
        num_paths=int(experiment["num_paths"]),
        metadata={"per_frame_mse_mean": experiment.get("per_frame_mse_mean")},
    )
