#!/usr/bin/env python3
"""MTPano / PD-BridgeNet CLI (Zhang et al., arXiv:2602.05330)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.mtpano.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.mtpano.config import CODE_URL, MTPanoConfig, PAPER_URL  # noqa: E402
from ltx_trainer.mtpano.datasets import training_dataset_card  # noqa: E402
from ltx_trainer.mtpano.paper import framework_card  # noqa: E402
from ltx_trainer.mtpano.pipeline import evaluation_demo_run  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(training_dataset_card(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.mtpano.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = MTPanoConfig(height=args.height, width=args.width)
    out = evaluation_demo_run(cfg, device=args.device)
    print(json.dumps({"paper": PAPER_URL, "code": CODE_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="MTPano multi-task panoramic dense prediction (arXiv:2602.05330)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--height", type=int, default=64)
    demo.add_argument("--width", type=int, default=128)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
