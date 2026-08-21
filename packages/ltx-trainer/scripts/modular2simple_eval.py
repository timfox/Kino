#!/usr/bin/env python3
"""Modular2Simple CLI (OpenSCENARIO modular scenarios)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.modular2simple.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.modular2simple.config import PAPER_REPO, Modular2SimpleConfig  # noqa: E402
from ltx_trainer.modular2simple.integration import integration_bundle  # noqa: E402
from ltx_trainer.modular2simple.paper import evaluation_demo, framework_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.modular2simple.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    print(json.dumps(integration_bundle(), indent=2))
    return 0


def _cmd_case(_: argparse.Namespace) -> int:
    from ltx_trainer.modular2simple.case_study import intersection_case_study_graph

    print(json.dumps(intersection_case_study_graph(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out = evaluation_demo(seed=args.seed)
    out["config"] = Modular2SimpleConfig().__dict__
    out["paper"] = PAPER_REPO
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Modular2Simple OpenSCENARIO modular scenario CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("plan")
    sub.add_parser("case")

    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    return {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "smoke": _cmd_smoke,
        "plan": _cmd_plan,
        "case": _cmd_case,
        "demo": _cmd_demo,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
