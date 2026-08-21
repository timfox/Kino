"""P2GS training and inference pipeline."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.p2gs.cameras import Camera, init_points_from_ply, load_image_chw, load_scene_cameras
from ltx_trainer.p2gs.gaussians import GaussianModel
from ltx_trainer.p2gs.losses import P2GSLossConfig, P2GSLosses, build_hdr_pairs, hdr_inconsistency_score, std_luminance
from ltx_trainer.p2gs.photometric import ViewPhotometricParams, render_ldr
from ltx_trainer.p2gs.rasterize import render_linear_hdr, render_linear_hdr_fast


@dataclass
class P2GSConfig:
    scene_dir: str
    output_dir: str
    iterations: int = 2000
    lr_gaussians: float = 1e-3
    lr_photo: float = 1e-2
    max_points: int = 5000
    max_gaussians_render: int = 512
    image_scale: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loss: P2GSLossConfig = field(default_factory=P2GSLossConfig)
    log_interval: int = 100
    use_fast_raster: bool = True
    seed: int = 42


def render_scene_ldr(
    gaussians: GaussianModel,
    view_params: ViewPhotometricParams,
    cam: Camera,
    view_idx: int = 0,
    *,
    exposure: float | None = None,
    gamma: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Render LDR + linear HDR for one view (optional override exposure/gamma)."""
    hdr = render_linear_hdr(gaussians, cam)
    if exposure is not None and gamma is not None:
        ldr = render_ldr(hdr, exposure=exposure, gamma=gamma)
    elif exposure is not None:
        ldr = render_ldr(hdr, exposure=exposure, gamma=view_params.gamma_view(view_idx))
    elif gamma is not None:
        ldr = render_ldr(hdr, exposure=view_params.exposure_view(view_idx), gamma=gamma)
    else:
        ldr = view_params.render_view(hdr, view_idx)
    return ldr, hdr


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[GaussianModel, ViewPhotometricParams, dict]:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    n_views = int(meta.get("num_views", ckpt["view_params"]["_log_exposure"].shape[0]))
    vp = ViewPhotometricParams(n_views).to(device)
    vp.load_state_dict(ckpt["view_params"])
    g = GaussianModel(
        ckpt["gaussians"]["_xyz"],
        ckpt["gaussians"]["_features"],
        scales=ckpt["gaussians"]["_scaling"],
        rotations=ckpt["gaussians"]["_rotation"],
        opacity=ckpt["gaussians"]["_opacity"],
    ).to(device)
    return g, vp, meta


class P2GSTrainer:
    def __init__(self, config: P2GSConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

    def _scaled_cameras(self, cameras: list[Camera]) -> list[Camera]:
        scale = self.config.image_scale
        if scale == 1.0:
            return cameras
        out: list[Camera] = []
        for c in cameras:
            out.append(
                Camera(
                    c.image_path,
                    max(1, int(c.width * scale)),
                    max(1, int(c.height * scale)),
                    c.fx * scale,
                    c.fy * scale,
                    c.cx * scale,
                    c.cy * scale,
                    c.R,
                    c.t,
                )
            )
        return out

    def _init_gaussians(self, scene: Path, cameras: list[Camera]) -> GaussianModel:
        pts = init_points_from_ply(scene, max_points=self.config.max_points)
        if pts is not None and pts.shape[0] > 0:
            if pts.shape[0] > self.config.max_points:
                idx = torch.randperm(pts.shape[0])[: self.config.max_points]
                pts = pts[idx]
            colors = torch.full((pts.shape[0], 3), 0.4)
            g = GaussianModel.from_point_cloud(pts, colors=colors, scale=0.06)
        else:
            centers = torch.stack([c.center for c in cameras]).mean(dim=0)
            g = GaussianModel.random_init(
                min(self.config.max_points, 800),
                centers,
                device=self.device,
            )
        return g.to(self.device)

    def train(self) -> Path:
        cfg = self.config
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        scene = Path(cfg.scene_dir).expanduser().resolve()
        out = Path(cfg.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)

        cameras = self._scaled_cameras(load_scene_cameras(scene))
        view_params = ViewPhotometricParams(len(cameras)).to(self.device)
        gaussians = self._init_gaussians(scene, cameras)

        loss_fn = P2GSLosses(cfg.loss)
        render_fn = render_linear_hdr_fast if cfg.use_fast_raster else render_linear_hdr

        params = [
            {"params": gaussians.parameters(), "lr": cfg.lr_gaussians},
            {"params": view_params.parameters(), "lr": cfg.lr_photo},
        ]
        opt = torch.optim.Adam(params)

        pairs = [(i, j) for i in range(len(cameras)) for j in range(i + 1, len(cameras))]
        if len(pairs) > 32:
            pairs = random.sample(pairs, 32)

        for it in range(1, cfg.iterations + 1):
            vi = random.randrange(len(cameras))
            vj = random.randrange(len(cameras))
            while vj == vi and len(cameras) > 1:
                vj = random.randrange(len(cameras))

            cam_i = cameras[vi]
            gt = load_image_chw(cam_i, scale=1.0, device=self.device)
            hdr_i = render_fn(gaussians, cam_i, max_gaussians=cfg.max_gaussians_render)
            pred = view_params.render_view(hdr_i, vi)

            cam_j = cameras[vj]
            hdr_j = render_fn(gaussians, cam_j, max_gaussians=cfg.max_gaussians_render)
            exp_pairs = build_hdr_pairs({vi: hdr_i, vj: hdr_j}, view_params, [(vi, vj)])

            loss, stats = loss_fn.total(pred, gt, exp_pairs, view_params)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(gaussians.parameters()) + list(view_params.parameters()), 1.0)
            opt.step()

            if it % cfg.log_interval == 0 or it == cfg.iterations:
                print(
                    f"[{it}/{cfg.iterations}] total={stats['l_total']:.4f} "
                    f"photo={stats['l_photo']:.4f} exp={stats['l_exp']:.4f} "
                    f"e_mean={view_params.exposure.mean().item():.3f}"
                )

        ckpt_path = out / "p2gs_checkpoint.pt"
        meta = {
            "scene_dir": str(scene),
            "num_views": len(cameras),
            "iterations": cfg.iterations,
            "paper": "arXiv:2605.16925",
            "image_scale": cfg.image_scale,
        }
        torch.save(
            {
                "meta": meta,
                "config": asdict(cfg),
                "gaussians": gaussians.state_dict(),
                "view_params": view_params.state_dict(),
            },
            ckpt_path,
        )
        (out / "p2gs_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Evaluation metrics on training views
        with torch.inference_mode():
            ldrs = []
            exps = []
            for idx, cam in enumerate(cameras):
                hdr = render_fn(gaussians, cam, max_gaussians=cfg.max_gaussians_render)
                ldr = view_params.render_view(hdr, idx)
                ldrs.append(ldr.cpu())
                exps.append(view_params.exposure[idx].cpu())
            stack = torch.stack(ldrs, dim=0)
            metrics = {
                "his": hdr_inconsistency_score(stack, torch.stack(exps)),
                "std_luminance": std_luminance(stack),
                "mean_exposure": float(view_params.exposure.mean()),
                "mean_gamma": float(view_params.gamma.mean()),
            }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {ckpt_path} metrics={metrics}")
        return out
