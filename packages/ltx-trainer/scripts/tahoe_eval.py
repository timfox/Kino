#!/usr/bin/env python3
"""Tahoe evaluation CLI (Chen et al., arXiv:2606.12387)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.tahoe.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.tahoe.config import TahoeConfig  # noqa: E402
from ltx_trainer.tahoe.integration import framework_card, integration_bundle  # noqa: E402
from ltx_trainer.tahoe.pipeline import evaluation_demo, evaluation_smoke  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(TahoeConfig()), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
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


def _cmd_query(args: argparse.Namespace) -> int:
    from ltx_trainer.tahoe.running_example import seed_hint_bank
    from ltx_trainer.tahoe.query_pipeline import run_hint_guided_query

    bank = seed_hint_bank()
    result = run_hint_guided_query(
        args.question,
        bank,
        database_id=args.database or None,
        user_id=args.user or None,
    )
    print(json.dumps(result.__dict__, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Tahoe Text-to-SQL Hint Bank evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("demo")
    sub.add_parser("plan")
    q = sub.add_parser("query", help="Run hint-guided query + syntax critic stub")
    q.add_argument("question", help="Natural language question")
    q.add_argument("--database", default="", help="Database scope id (e.g. GA4)")
    q.add_argument("--user", default="", help="User scope id")
    args = p.parse_args()
    return {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "smoke": _cmd_smoke,
        "demo": _cmd_demo,
        "plan": _cmd_plan,
        "query": _cmd_query,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
