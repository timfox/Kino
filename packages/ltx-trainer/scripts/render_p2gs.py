#!/usr/bin/env python3
"""Render exposure / gamma sweeps from a trained P2GS checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from ltx_trainer.p2gs.cameras import load_scene_cameras
from ltx_trainer.p2gs.pipeline import load_checkpoint, render_scene_ldr


def _save_chw(path: Path, tensor: torch.Tensor) -> None:
    x = tensor.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render P2GS checkpoint")
    p.add_argument("checkpoint", type=Path, help="p2gs_checkpoint.pt")
    p.add_argument("scene_dir", type=Path, help="Same cameras.json scene as training")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--view", type=int, default=0, help="Camera index")
    p.add_argument(
        "--exposure-sweep",
        default=None,
        help="Comma-separated exposure scales (e.g. 0.5,1.0,1.5)",
    )
    p.add_argument("--gamma", type=float, default=None, help="Override gamma (default: view mean)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    gaussians, vp, meta = load_checkpoint(args.checkpoint, device=args.device)
    cameras = load_scene_cameras(args.scene_dir)
    scale = float(meta.get("image_scale", 1.0))
    if scale != 1.0:
        from ltx_trainer.p2gs.cameras import Camera

        c = cameras[args.view]
        cameras = [
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
        ]
        view_idx = 0
    else:
        view_idx = args.view

    cam = cameras[view_idx if scale == 1.0 else 0]
    e_mean, g_mean = vp.mean_render_params()
    gamma = args.gamma if args.gamma is not None else float(g_mean)

    if args.exposure_sweep:
        exposures = [float(x.strip()) for x in args.exposure_sweep.split(",") if x.strip()]
    else:
        exposures = [float(e_mean)]

    for e in exposures:
        ldr, hdr = render_scene_ldr(gaussians, vp, cam, view_idx, exposure=e, gamma=gamma)
        tag = f"e{e:.2f}_g{gamma:.2f}".replace(".", "p")
        _save_chw(args.output / f"ldr_{tag}.png", ldr)
        hdr_vis = (hdr / hdr.quantile(0.99).clamp(min=1e-6)).clamp(0, 1).pow(1 / gamma)
        _save_chw(args.output / f"hdr_tonemap_{tag}.png", hdr_vis)
        print(f"Wrote {args.output / f'ldr_{tag}.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
