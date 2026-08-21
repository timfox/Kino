#!/usr/bin/env python3
"""PriOr-Flow CLI (Liu et al., arXiv:2506.23897)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.prior_flow.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.prior_flow.config import PAPER_URL, PriorFlowConfig  # noqa: E402
from ltx_trainer.prior_flow.datasets import datasets_card  # noqa: E402
from ltx_trainer.prior_flow.paper import framework_card  # noqa: E402
from ltx_trainer.prior_flow.pipeline import (  # noqa: E402
    ablation_modules,
    distortion_demo,
    evaluation_demo_run,
    orthogonal_demo,
    train_step,
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
    from ltx_trainer.prior_flow.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_ablation(args: argparse.Namespace) -> int:
    from ltx_trainer.prior_flow.benchmarks import TABLE1_ABLATION

    cfg = PriorFlowConfig(height=args.height, width=args.width, num_iterations=args.iters)
    print(
        json.dumps(
            {"table1_paper": TABLE1_ABLATION, "stub_forward": ablation_modules(cfg)},
            indent=2,
        )
    )
    return 0


def _cmd_distortion(args: argparse.Namespace) -> int:
    print(json.dumps(distortion_demo(args.height, args.width), indent=2))
    return 0


def _cmd_orthogonal(args: argparse.Namespace) -> int:
    cfg = PriorFlowConfig(height=args.height, width=args.width)
    print(json.dumps(orthogonal_demo(cfg), indent=2))
    return 0


def _cmd_train_stub(args: argparse.Namespace) -> int:
    cfg = PriorFlowConfig(height=args.height, width=args.width, num_iterations=args.iters)
    print(json.dumps(train_step(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = PriorFlowConfig(height=args.height, width=args.width, num_iterations=args.iters)
    print(json.dumps({"paper": PAPER_URL, **evaluation_demo_run(cfg)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="PriOr-Flow panoramic optical flow stub")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("tables", _cmd_tables),
        ("dataset", _cmd_dataset),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)
    for name, fn in (
        ("demo", _cmd_demo),
        ("ablation", _cmd_ablation),
        ("distortion", _cmd_distortion),
        ("orthogonal", _cmd_orthogonal),
        ("train-stub", _cmd_train_stub),
    ):
        sp = sub.add_parser(name)
        sp.add_argument("--height", type=int, default=64)
        sp.add_argument("--width", type=int, default=128)
        sp.add_argument("--iters", type=int, default=3)
        sp.set_defaults(func=fn)
    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
