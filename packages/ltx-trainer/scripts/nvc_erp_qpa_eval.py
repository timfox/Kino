#!/usr/bin/env python3
"""NVC ERP QPA CLI (Arai et al., arXiv:2512.20093)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.nvc_erp_qpa.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.nvc_erp_qpa.config import PAPER_URL, NvcErpQpaConfig  # noqa: E402
from ltx_trainer.nvc_erp_qpa.datasets import datasets_card  # noqa: E402
from ltx_trainer.nvc_erp_qpa.paper import framework_card  # noqa: E402
from ltx_trainer.nvc_erp_qpa.pipeline import evaluation_demo_run, interpolation_ablation  # noqa: E402


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
    from ltx_trainer.nvc_erp_qpa.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_interp(_: argparse.Namespace) -> int:
    cfg = NvcErpQpaConfig(erp_height=args.height, erp_width=args.width)
    print(json.dumps(interpolation_ablation(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = NvcErpQpaConfig(erp_height=args.height, erp_width=args.width)
    print(json.dumps({"paper": PAPER_URL, **evaluation_demo_run(cfg)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="NVC ERP quality-parameter adaptation (DCVC-RT)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    interp = sub.add_parser("interp")
    interp.add_argument("--height", type=int, default=64)
    interp.add_argument("--width", type=int, default=128)
    interp.set_defaults(func=_cmd_interp)
    demo = sub.add_parser("demo")
    demo.add_argument("--height", type=int, default=64)
    demo.add_argument("--width", type=int, default=128)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
