#!/usr/bin/env python3
"""360Anything CLI (Wu et al., arXiv:2601.16192)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.anything360.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.anything360.config import PAPER_URL, PROJECT_URL, Anything360Config  # noqa: E402
from ltx_trainer.anything360.datasets import datasets_card  # noqa: E402
from ltx_trainer.anything360.paper import framework_card  # noqa: E402
from ltx_trainer.anything360.pipeline import cle_ablation_demo, evaluation_demo_run  # noqa: E402


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
    from ltx_trainer.anything360.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_cle(_: argparse.Namespace) -> int:
    cfg = Anything360Config(erp_height=args.height, erp_width=args.width)
    print(json.dumps(cle_ablation_demo(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = Anything360Config(erp_height=args.height, erp_width=args.width)
    out = evaluation_demo_run(cfg)
    print(json.dumps({"paper": PAPER_URL, "project": PROJECT_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="360Anything geometry-free pers→360°")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    cle = sub.add_parser("cle")
    cle.add_argument("--height", type=int, default=64)
    cle.add_argument("--width", type=int, default=128)
    cle.set_defaults(func=_cmd_cle)
    demo = sub.add_parser("demo")
    demo.add_argument("--height", type=int, default=64)
    demo.add_argument("--width", type=int, default=128)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
