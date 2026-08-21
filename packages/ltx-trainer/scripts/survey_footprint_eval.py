#!/usr/bin/env python3
"""Survey Footprint Explorer CLI (Ahad et al. arXiv:2605.11099)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.survey_footprint.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.survey_footprint.config import TOOL_URL, SurveyFootprintConfig  # noqa: E402
from ltx_trainer.survey_footprint.paper import framework_card  # noqa: E402
from ltx_trainer.survey_footprint.pipeline import evaluation_demo_run  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.survey_footprint.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = SurveyFootprintConfig(mc_samples=args.samples)
    out = evaluation_demo_run(cfg)
    print(json.dumps({"tool": TOOL_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Survey Footprint Explorer (arXiv:2605.11099)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--samples", type=int, default=5000, help="MOC Monte Carlo samples")
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
