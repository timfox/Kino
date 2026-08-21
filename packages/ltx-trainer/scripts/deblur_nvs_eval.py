#!/usr/bin/env python3
"""DeblurNVS blur-aware NVS CLI (Shi et al. arXiv:2606.01315)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.deblur_nvs.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.deblur_nvs.config import DeblurNVSConfig, PAPER_URL  # noqa: E402
from ltx_trainer.deblur_nvs.dataset import dataset_card  # noqa: E402
from ltx_trainer.deblur_nvs.paper import framework_card  # noqa: E402
from ltx_trainer.deblur_nvs.pipeline import evaluation_demo_run, infer_novel_view  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(dataset_card(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.deblur_nvs.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    from ltx_trainer.deblur_nvs.integration import ltx_plan_stub

    print(json.dumps(ltx_plan_stub(), indent=2))
    return 0


def _cmd_blur_pair(args: argparse.Namespace) -> int:
    import torch

    from ltx_trainer.deblur_nvs.blur_synthesis import synthesize_blur_pair

    sharp = torch.rand(3, args.height, args.width)
    s, b, n = synthesize_blur_pair(sharp, seed=args.seed)
    print(
        json.dumps(
            {
                "window": n,
                "sharp_shape": list(s.shape),
                "blur_shape": list(b.shape),
                "mean_abs_delta": float((s - b).abs().mean().item()),
            },
            indent=2,
        )
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = DeblurNVSConfig(height=args.height, width=args.width, context_views=args.views)
    out = evaluation_demo_run(cfg, device=args.device, seed=args.seed)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    import torch

    from ltx_trainer.deblur_nvs.latent_models import DeblurNVSStub
    from ltx_trainer.deblur_nvs.synthetic import synthetic_views

    cfg = DeblurNVSConfig(height=args.height, width=args.width, context_views=args.views)
    dev = torch.device(args.device)
    batch = synthetic_views(cfg, device=dev, seed=args.seed)
    model = DeblurNVSStub(cfg).to(dev)
    out = infer_novel_view(model, batch["blur_context"], batch["camera"])
    print(
        json.dumps(
            {
                "paper": PAPER_URL,
                "novel_view_shape": list(out["novel_view"].shape),
                "context_views": cfg.context_views,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="DeblurNVS evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("dataset")
    sub.add_parser("smoke")
    sub.add_parser("plan")

    blur_pair = sub.add_parser("blur-pair")
    blur_pair.add_argument("--seed", type=int, default=0)
    blur_pair.add_argument("--height", type=int, default=48)
    blur_pair.add_argument("--width", type=int, default=80)

    demo = sub.add_parser("demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--height", type=int, default=56)
    demo.add_argument("--width", type=int, default=96)
    demo.add_argument("--views", type=int, default=3)

    infer = sub.add_parser("infer")
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("--height", type=int, default=56)
    infer.add_argument("--width", type=int, default=96)
    infer.add_argument("--views", type=int, default=3)

    args = p.parse_args()
    handlers = {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "dataset": _cmd_dataset,
        "smoke": _cmd_smoke,
        "plan": _cmd_plan,
        "blur-pair": _cmd_blur_pair,
        "demo": _cmd_demo,
        "infer": _cmd_infer,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
