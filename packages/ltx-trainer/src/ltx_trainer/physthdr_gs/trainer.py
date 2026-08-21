"""Full PhysHDR-GS training loop (multi-view + dual branch + I-GS densify)."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.physthdr_gs.cameras import (
    Camera,
    camera_to_device,
    init_points_from_ply,
    load_image_chw,
    load_scene_cameras,
    scale_cameras,
    synthetic_camera_rig,
)
from ltx_trainer.physthdr_gs.config import EXPOSURE_TIMES, LDR_OE_INDICES, PhysHDRConfig
from ltx_trainer.physthdr_gs.densify import accumulate_viewspace_grad, densify_step
from ltx_trainer.physthdr_gs.gaussians import HDRGaussianField
from ltx_trainer.physthdr_gs.losses import PhysHDRLoss, PhysHDRLossConfig
from ltx_trainer.physthdr_gs.model import PhysHDRGS
from ltx_trainer.physthdr_gs.synthetic import ldr_from_hdr, tone_map_mu_law


def _sample_exposure(cfg: PhysHDRConfig, rng: random.Random) -> float:
    if cfg.exposure_setting == "exp1":
        return EXPOSURE_TIMES[LDR_OE_INDICES[1]]
    idx = rng.choice(LDR_OE_INDICES)
    return EXPOSURE_TIMES[idx]


def _synthetic_gt_for_camera(
    seed: int,
    cam: Camera,
    exposure: float,
    device: torch.device,
) -> Tensor:
    """Procedural HDR scene GT LDR for synthetic rig training."""
    from ltx_trainer.physthdr_gs.synthetic import _make_scene

    h, w = cam.height, cam.width
    hdr, _ = _make_scene(min(h, w), seed)
    if hdr.shape[-1] != h or hdr.shape[-2] != w:
        hdr = torch.nn.functional.interpolate(
            hdr.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    ldr = ldr_from_hdr(hdr, exposure)
    return ldr.unsqueeze(0).to(device)


class PhysHDRTrainer:
    def __init__(self, cfg: PhysHDRConfig | None = None) -> None:
        self.cfg = cfg or PhysHDRConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_cameras(self) -> list[Camera]:
        if self.cfg.scene_dir:
            scene = Path(self.cfg.scene_dir).expanduser().resolve()
            cams = load_scene_cameras(scene)
            return scale_cameras(cams, self.cfg.image_scale)
        size = self.cfg.image_size
        return synthetic_camera_rig(
            self.cfg.num_train_views,
            size,
            size,
            device=self.device,
        )

    def _init_field(self, scene: Path | None, cameras: list[Camera]) -> HDRGaussianField:
        if scene is not None:
            pts = init_points_from_ply(scene, max_points=self.cfg.max_gaussians)
            if pts is not None and pts.shape[0] > 0:
                return HDRGaussianField.from_point_cloud(
                    pts.to(self.device),
                    max_points=self.cfg.max_gaussians,
                ).to(self.device)
        centers = torch.stack([c.center for c in cameras]).mean(dim=0).to(self.device)
        field = HDRGaussianField(self.cfg.num_gaussians, init_radius=self.cfg.init_radius)
        field._max_points = self.cfg.max_gaussians
        field = field.to(self.device)
        with torch.no_grad():
            field._xyz.copy_(centers + torch.randn_like(field._xyz) * 0.5)
        return field

    def train(self, output_dir: str | Path) -> Path:
        cfg = self.cfg
        rng = random.Random(42)
        torch.manual_seed(42)
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        scene = Path(cfg.scene_dir).expanduser().resolve() if cfg.scene_dir else None

        cameras = self._load_cameras()
        field = self._init_field(scene, cameras)
        model = PhysHDRGS(cfg, field=field).to(self.device)

        loss_fn = PhysHDRLoss(
            PhysHDRLossConfig(
                lambda_rec=cfg.lambda_rec,
                lambda_cons=cfg.lambda_cons,
                lambda_unit=cfg.lambda_unit,
                gamma_mse=cfg.gamma_mse,
            )
        )
        opt = torch.optim.Adam(
            [
                {"params": model.field.parameters(), "lr": cfg.lr_gaussians},
                {"params": model.composer.parameters(), "lr": cfg.lr_radiance},
                {"params": model.modulator.parameters(), "lr": cfg.lr_radiance},
                {"params": model.tone_mapper.parameters(), "lr": cfg.lr_tonemap},
            ]
        )
        grad_state: dict[str, Tensor] | None = None
        log: list[dict[str, float]] = []

        for it in range(cfg.max_iterations):
            model.set_iteration(it)
            model._train_iter = it
            vi = rng.randrange(len(cameras))
            cam = camera_to_device(cameras[vi], self.device)
            exp = _sample_exposure(cfg, rng)
            t = torch.tensor(exp, device=self.device, dtype=torch.float32)

            if scene and cam.image_path.is_file():
                target = load_image_chw(cam, scale=1.0, device=self.device).unsqueeze(0)
            else:
                target = _synthetic_gt_for_camera(vi * 1000 + it, cam, exp, self.device)

            opt.zero_grad(set_to_none=True)
            out_fwd = model(t, lighting_level=t, camera=cam if cfg.use_perspective else None)
            preds = {"ildr": out_fwd.ildr, "iig": out_fwd.iig, "igi": out_fwd.igi}
            loss, stats = loss_fn(
                preds=preds,
                target_ldr=target,
                ihdr_scaled=out_fwd.ihdr_scaled,
                ihdr_relit=out_fwd.ihdr_relit,
                ihdr=out_fwd.ihdr,
                exposure=exp,
                use_cons=cfg.use_hdr_cons and cfg.use_gi_branch,
            )
            loss.backward()

            if (
                cfg.use_perspective
                and out_fwd.viewspace is not None
                and it >= cfg.densify_from_iter
                and it % cfg.densify_interval == 0
            ):
                sa = None
                if cfg.use_igs and cfg.use_gi_branch:
                    from ltx_trainer.physthdr_gs.gradient_scaling import scaling_factor

                    sa = scaling_factor(out_fwd.la, out_fwd.la_hat, s=cfg.scale_s)
                grad_state = accumulate_viewspace_grad(model.field, out_fwd.viewspace, sa, state=grad_state)
                if it > 0 and it % (cfg.densify_interval * 5) == 0:
                    n_before = model.field.num_points
                    cloned = densify_step(
                        model.field,
                        grad_state,
                        tau=cfg.densify_tau,
                        scale_s=cfg.scale_s,
                        la=out_fwd.la.detach(),
                        la_hat=out_fwd.la_hat.detach(),
                    )
                    stats["densified"] = float(cloned)
                    if model.field.num_points != n_before:
                        opt = torch.optim.Adam(
                            [
                                {"params": model.field.parameters(), "lr": cfg.lr_gaussians},
                                {"params": model.composer.parameters(), "lr": cfg.lr_radiance},
                                {"params": model.modulator.parameters(), "lr": cfg.lr_radiance},
                                {"params": model.tone_mapper.parameters(), "lr": cfg.lr_tonemap},
                            ]
                        )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            stats["step"] = float(it)
            stats["num_gaussians"] = float(model.field.num_points)
            log.append(stats)
            if it % cfg.log_interval == 0 or it == cfg.max_iterations - 1:
                print(
                    f"[{it}/{cfg.max_iterations}] loss={stats['loss_total']:.4f} "
                    f"rec={stats['loss_rec']:.4f} cons={stats['loss_cons']:.4f} "
                    f"N={model.field.num_points}"
                )

        ckpt = out / "physthdr_gs.pt"
        meta = {
            "config": {k: v for k, v in asdict(cfg).items()},
            "num_views": len(cameras),
            "scene_dir": str(scene) if scene else None,
            "paper": "arXiv:2603.28020",
        }
        torch.save({"meta": meta, "state_dict": model.state_dict(), "field": model.field.state_dict()}, ckpt)
        (out / "train_log.json").write_text(json.dumps(log[-50:], indent=2), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out
