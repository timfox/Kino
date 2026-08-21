#!/usr/bin/env python3
"""MDDN ODISR CLI (Yang et al., arXiv:2512.17343)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.mddn.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.mddn.config import PAPER_URL, MddnConfig  # noqa: E402
from ltx_trainer.mddn.datasets import datasets_card  # noqa: E402
from ltx_trainer.mddn.paper import framework_card  # noqa: E402
from ltx_trainer.mddn.pipeline import evaluation_demo_run, fusion_ablation  # noqa: E402


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
    from ltx_trainer.mddn.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_fusion(_: argparse.Namespace) -> int:
    cfg = MddnConfig(erp_height=args.height, erp_width=args.width)
    print(json.dumps(fusion_ablation(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = MddnConfig(erp_height=args.height, erp_width=args.width, scale=args.scale)
    print(json.dumps({"paper": PAPER_URL, **evaluation_demo_run(cfg)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="MDDN omnidirectional image super-resolution")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    fus = sub.add_parser("fusion")
    fus.add_argument("--height", type=int, default=32)
    fus.add_argument("--width", type=int, default=64)
    fus.set_defaults(func=_cmd_fusion)
    demo = sub.add_parser("demo")
    demo.add_argument("--height", type=int, default=32)
    demo.add_argument("--width", type=int, default=64)
    demo.add_argument("--scale", type=int, default=4)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
