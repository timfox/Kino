#!/usr/bin/env python3
"""Gimbal360 panoramic completion CLI (Lu et al. arXiv:2603.23179)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.gimbal360.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.gimbal360.config import PAPER_URL, Gimbal360Config  # noqa: E402
from ltx_trainer.gimbal360.paper import framework_card  # noqa: E402
from ltx_trainer.gimbal360.horizon360 import dataset_card  # noqa: E402
from ltx_trainer.gimbal360.inference import SamplerConfig, complete_panorama_stub  # noqa: E402
from ltx_trainer.gimbal360.pipeline import evaluation_demo_run  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.gimbal360.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(dataset_card(), indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    import torch

    from ltx_trainer.gimbal360.gimbal360_net import Gimbal360CompletionStub
    from ltx_trainer.gimbal360.synthetic import synthetic_batch

    cfg = Gimbal360Config(
        erp_height=args.erp_height,
        erp_width=args.erp_width,
        perspective_height=args.perspective_size,
        perspective_width=args.perspective_size,
    )
    dev = torch.device(args.device)
    batch = synthetic_batch(cfg, batch_size=1, device=dev)
    model = Gimbal360CompletionStub(cfg).to(dev)
    sampler = SamplerConfig(num_steps=args.steps, cfg_scale=args.cfg_scale, step_size=args.step_size)
    out = complete_panorama_stub(model, batch["perspective"], batch["mask"], cfg=cfg, sampler=sampler)
    print(
        json.dumps(
            {
                "paper": PAPER_URL,
                "erp_shape": list(out["erp"].shape),
                "latent_shape": list(out["latent"].shape),
                "total_azimuth_shift": int(out["total_azimuth_shift"].item()),
                "denoise_steps": int(out["steps"].item()),
            },
            indent=2,
        )
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = Gimbal360Config(
        erp_height=args.erp_height,
        erp_width=args.erp_width,
        perspective_height=args.perspective_size,
        perspective_width=args.perspective_size,
    )
    out = evaluation_demo_run(cfg, device=args.device)
    print(json.dumps({"paper": PAPER_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Gimbal360 ERP completion (arXiv:2603.23179)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    infer = sub.add_parser("infer", help="Shift-equivariant completion stub (Appendix A.2)")
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--erp-height", type=int, default=48)
    infer.add_argument("--erp-width", type=int, default=96)
    infer.add_argument("--perspective-size", type=int, default=32)
    infer.add_argument("--steps", type=int, default=6)
    infer.add_argument("--cfg-scale", type=float, default=1.0)
    infer.add_argument("--step-size", type=float, default=0.1)
    infer.set_defaults(func=_cmd_infer)
    demo = sub.add_parser("demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--erp-height", type=int, default=48)
    demo.add_argument("--erp-width", type=int, default=96)
    demo.add_argument("--perspective-size", type=int, default=32)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
