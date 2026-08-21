#!/usr/bin/env python3
"""LLMs+Graphs tutorial CLI (Khan et al., arXiv:2606.11560)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.llms_graphs.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.llms_graphs.config import PAPER_URL, LLMsGraphsConfig  # noqa: E402
from ltx_trainer.llms_graphs.integration import integration_bundle  # noqa: E402
from ltx_trainer.llms_graphs.paper import evaluation_demo, framework_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.llms_graphs.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    print(json.dumps(integration_bundle(), indent=2))
    return 0


def _cmd_synergies(_: argparse.Namespace) -> int:
    from ltx_trainer.llms_graphs.synergies import synergy_summary

    print(json.dumps(synergy_summary(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out = evaluation_demo(seed=args.seed)
    out["config"] = LLMsGraphsConfig().__dict__
    out["paper"] = PAPER_URL
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="LLMs+Graphs tutorial evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("plan")
    sub.add_parser("synergies")

    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    return {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "smoke": _cmd_smoke,
        "plan": _cmd_plan,
        "synergies": _cmd_synergies,
        "demo": _cmd_demo,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
