#!/usr/bin/env python3
"""GenPCR CLI (Jiang et al., arXiv:2512.09407)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.genpcr.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.genpcr.config import PAPER_URL, GenPcrConfig  # noqa: E402
from ltx_trainer.genpcr.datasets import datasets_card  # noqa: E402
from ltx_trainer.genpcr.paper import framework_card  # noqa: E402
from ltx_trainer.genpcr.pipeline import evaluation_demo_run  # noqa: E402


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
    from ltx_trainer.genpcr.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_ablation(_: argparse.Namespace) -> int:
    from ltx_trainer.genpcr.benchmarks import THREEDMATCH_ABLATION_FCGF_SD

    print(json.dumps(THREEDMATCH_ABLATION_FCGF_SD, indent=2))
    return 0


def _cmd_theory(_: argparse.Namespace) -> int:
    from ltx_trainer.genpcr.theory import coupled_elbo_summary

    print(json.dumps(coupled_elbo_summary(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    mode = args.mode
    cfg = GenPcrConfig(mode=mode, image_height=args.height, image_width=args.width)
    print(json.dumps({"paper": PAPER_URL, **evaluation_demo_run(cfg)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Generative point cloud registration")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    sub.add_parser("ablation").set_defaults(func=_cmd_ablation)
    sub.add_parser("theory").set_defaults(func=_cmd_theory)
    demo = sub.add_parser("demo")
    demo.add_argument("--mode", choices=("depth", "lidar"), default="depth")
    demo.add_argument("--height", type=int, default=32)
    demo.add_argument("--width", type=int, default=32)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
