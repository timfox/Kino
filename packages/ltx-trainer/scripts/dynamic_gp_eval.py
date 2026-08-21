#!/usr/bin/env python3
"""Dynamic Gaussian Process CLI (van Hulst et al., arXiv:2606.06705)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.dynamic_gp.basis import morphology_card  # noqa: E402
from ltx_trainer.dynamic_gp.benchmarks import heat_error_curve, summary_anchors, wave_example_card  # noqa: E402
from ltx_trainer.dynamic_gp.config import DynamicGPConfig  # noqa: E402
from ltx_trainer.dynamic_gp.error_analysis import error_card  # noqa: E402
from ltx_trainer.dynamic_gp.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.dynamic_gp.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.dynamic_gp.pipeline import run_demo, run_wave_demo  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {"heat_error_curve": heat_error_curve(), "wave_example": wave_example_card(), "summary": summary_anchors()},
            indent=2,
        )
    )
    return 0


def _cmd_method(_: argparse.Namespace) -> int:
    print(json.dumps({"separable_kernels": morphology_card(), "error_analysis": error_card()}, indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = DynamicGPConfig(num_basis=args.M, num_steps=args.steps)
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_wave(_: argparse.Namespace) -> int:
    print(json.dumps(run_wave_demo(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dynamic Gaussian Process evolving-function estimation stub")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("paper", _cmd_paper),
        ("tables", _cmd_tables),
        ("method", _cmd_method),
        ("wave", _cmd_wave),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    demo = sub.add_parser("demo")
    demo.add_argument("--M", type=int, default=31)
    demo.add_argument("--steps", type=int, default=5)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
