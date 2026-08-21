#!/usr/bin/env python3
"""PhysHDR-GS train / eval / table CLI (Zeng et al. arXiv:2603.28020)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402

from ltx_trainer.physthdr_gs.config import PAPER_URL, PROJECT_URL, PhysHDRConfig  # noqa: E402
from ltx_trainer.physthdr_gs.dataset import dataset_summary  # noqa: E402
from ltx_trainer.physthdr_gs.losses import PhysHDRLoss, PhysHDRLossConfig  # noqa: E402
from ltx_trainer.physthdr_gs.model import PhysHDRGS  # noqa: E402
from ltx_trainer.physthdr_gs.pipeline import (  # noqa: E402
    evaluate_psnr,
    paper_report,
    save_checkpoint,
    train_step,
)
from ltx_trainer.physthdr_gs.synthetic import sample_training_pair, synthesize_view  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "name": "PhysHDR-GS",
                "paper": "Physically Inspired Gaussian Splatting for HDR Novel View Synthesis",
                "url": PAPER_URL,
                "project": PROJECT_URL,
                "branches": ["image_exposure_ie", "gaussian_illumination_gi"],
                "losses": ["Lrec", "Lcons", "Lunit"],
                "components": ["radiance_composer", "illumination_modulator", "tone_mapper", "igs"],
                "datasets": list(dataset_summary()["datasets"].keys()),
            },
            indent=2,
        )
    )
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(paper_report(), indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = PhysHDRConfig(
        num_gaussians=args.gaussians,
        image_size=args.size,
        max_iterations=args.steps,
        lambda_cons=args.lambda_cons,
        use_gi_branch=not args.no_gi,
        use_hdr_cons=not args.no_cons,
        use_igs=not args.no_igs,
        use_perspective=not args.orthographic,
        num_train_views=args.views,
        scene_dir=args.scene_dir,
        log_interval=max(1, args.steps // 10),
    )
    if args.device != "cpu":
        import os

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from ltx_trainer.physthdr_gs.trainer import PhysHDRTrainer

    out = PhysHDRTrainer(cfg).train(Path(args.output_dir).expanduser().resolve())
    summary_path = out / "train_log.json"
    print(json.dumps({"output": str(out), "log": str(summary_path)}, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = PhysHDRConfig(image_size=args.size, num_gaussians=args.gaussians)
    model = PhysHDRGS(cfg).to(device)
    target, _, exp = sample_training_pair(args.size, seed=args.seed)
    target = target.to(device)
    loss_fn = PhysHDRLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for step in range(args.steps):
        model._train_iter = step
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(model, loss_fn, target_ldr=target, exposure=exp)
        loss.backward()
        opt.step()
    score = evaluate_psnr(model, target, exposure=exp)
    print(json.dumps({"psnr": score, "paper_hdr_syn_exp3_ours_dagger": 39.21, **stats}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="PhysHDR-GS HDR-NVS")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("knowledge")
    sp.set_defaults(func=_cmd_knowledge)

    sp = sub.add_parser("tables")
    sp.set_defaults(func=_cmd_tables)

    sp = sub.add_parser("train")
    sp.add_argument("-o", "--output-dir", default="physthdr_gs_train")
    sp.add_argument("--steps", type=int, default=100)
    sp.add_argument("--size", type=int, default=64)
    sp.add_argument("--gaussians", type=int, default=256)
    sp.add_argument("--lambda-cons", type=float, default=0.5)
    sp.add_argument("--lambda-unit", type=float, default=0.0)
    sp.add_argument("--no-gi", action="store_true")
    sp.add_argument("--no-cons", action="store_true")
    sp.add_argument("--no-igs", action="store_true")
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--orthographic", action="store_true", help="legacy orthographic splat (tests)")
    sp.add_argument("--views", type=int, default=8, help="synthetic training views")
    sp.add_argument("--scene-dir", default=None, help="SfM scene with cameras.json + images")
    sp.set_defaults(func=_cmd_train)

    sp = sub.add_parser("eval")
    sp.add_argument("--steps", type=int, default=50)
    sp.add_argument("--size", type=int, default=64)
    sp.add_argument("--gaussians", type=int, default=128)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_eval)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
