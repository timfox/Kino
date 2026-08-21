#!/usr/bin/env python3
"""Train P2GS on an SfM scene (Shimomura et al. arXiv:2605.16925)."""

from __future__ import annotations

import argparse

from ltx_trainer.p2gs.losses import P2GSLossConfig
from ltx_trainer.p2gs.pipeline import P2GSConfig, P2GSTrainer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train P2GS exposure-invariant HDR 3DGS")
    p.add_argument("scene_dir", help="SfM export with cameras.json + images/")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--max-points", type=int, default=5000)
    p.add_argument("--max-gaussians-render", type=int, default=512)
    p.add_argument("--image-scale", type=float, default=0.5, help="Resize factor for training")
    p.add_argument("--lr-gaussians", type=float, default=1e-3)
    p.add_argument("--lr-photo", type=float, default=1e-2)
    p.add_argument("--lambda-exp", type=float, default=0.01)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full-raster", action="store_true", help="Use slower per-Gaussian splat loop")
    args = p.parse_args(argv)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = P2GSConfig(
        scene_dir=args.scene_dir,
        output_dir=args.output,
        iterations=args.iterations,
        max_points=args.max_points,
        max_gaussians_render=args.max_gaussians_render,
        image_scale=args.image_scale,
        lr_gaussians=args.lr_gaussians,
        lr_photo=args.lr_photo,
        device=device,
        seed=args.seed,
        use_fast_raster=not args.full_raster,
        loss=P2GSLossConfig(lambda_exp=args.lambda_exp),
    )
    P2GSTrainer(cfg).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
