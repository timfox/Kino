#!/usr/bin/env python3
"""S3PO CLI (Baniya et al., arXiv:2506.14803)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.s3po.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.s3po.config import PAPER_URL, S3POConfig  # noqa: E402
from ltx_trainer.s3po.datasets import datasets_card  # noqa: E402
from ltx_trainer.s3po.paper import framework_card  # noqa: E402
from ltx_trainer.s3po.pipeline import (  # noqa: E402
    ablation_modules,
    evaluation_demo_run,
    train_step,
    wss_weights_demo,
)


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(datasets_card(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.s3po.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_ablation(args: argparse.Namespace) -> int:
    from ltx_trainer.s3po.benchmarks import TABLE5_360_SPECIFIC

    cfg = S3POConfig(lr_height=args.height, lr_width=args.width)
    print(
        json.dumps(
            {"table5_paper": TABLE5_360_SPECIFIC, "stub_forward": ablation_modules(cfg)},
            indent=2,
        )
    )
    return 0


def _cmd_wss(args: argparse.Namespace) -> int:
    print(json.dumps(wss_weights_demo(args.height, args.width), indent=2))
    return 0


def _cmd_train_stub(args: argparse.Namespace) -> int:
    cfg = S3POConfig(lr_height=args.height, lr_width=args.width)
    print(json.dumps(train_step(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = S3POConfig(lr_height=args.height, lr_width=args.width)
    print(json.dumps(evaluation_demo_run(cfg), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=f"S3PO 360° VSR — {PAPER_URL}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--height", type=int, default=48)
    demo.add_argument("--width", type=int, default=64)
    demo.set_defaults(func=_cmd_demo)
    ab = sub.add_parser("ablation")
    ab.add_argument("--height", type=int, default=32)
    ab.add_argument("--width", type=int, default=48)
    ab.set_defaults(func=_cmd_ablation)
    wss = sub.add_parser("wss-weights")
    wss.add_argument("--height", type=int, default=360)
    wss.add_argument("--width", type=int, default=480)
    wss.set_defaults(func=_cmd_wss)
    tr = sub.add_parser("train-stub")
    tr.add_argument("--height", type=int, default=32)
    tr.add_argument("--width", type=int, default=48)
    tr.set_defaults(func=_cmd_train_stub)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
