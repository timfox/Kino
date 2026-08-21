#!/usr/bin/env python3
"""NRP evaluation CLI (Soeteman et al., arXiv:2606.11946)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.nrp.benchmarks import TABLE_I  # noqa: E402
from ltx_trainer.nrp.config import NrpConfig  # noqa: E402
from ltx_trainer.nrp.integration import framework_card, integration_bundle  # noqa: E402
from ltx_trainer.nrp.pipeline import evaluation_demo, evaluation_smoke  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(NrpConfig()), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps({"table_i": TABLE_I}, indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo(), indent=2))
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    print(json.dumps(integration_bundle(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Neuro-Relational Programs evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("demo")
    sub.add_parser("plan")
    args = p.parse_args()
    return {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "smoke": _cmd_smoke,
        "demo": _cmd_demo,
        "plan": _cmd_plan,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
