#!/usr/bin/env python3
"""FairLend evaluation CLI (Rathod et al., arXiv:2606.12435)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.fairlend.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.fairlend.config import FairLendConfig  # noqa: E402
from ltx_trainer.fairlend.integration import framework_card, integration_bundle  # noqa: E402
from ltx_trainer.fairlend.pipeline import evaluation_demo, evaluation_smoke  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(FairLendConfig()), indent=2))
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


def _cmd_audit(_: argparse.Namespace) -> int:
    from ltx_trainer.fairlend.audit_pipeline import run_fair_lending_audit

    result = run_fair_lending_audit(FairLendConfig())
    print(json.dumps(result.__dict__, indent=2))
    return 0


def _cmd_binning(_: argparse.Namespace) -> int:
    from ltx_trainer.fairlend.binning import demo_standard_vs_fair_income
    from ltx_trainer.fairlend.epsilon_sweep import sweep_summary_paper_aligned

    print(
        json.dumps(
            {
                "standard_vs_fair": demo_standard_vs_fair_income(),
                "epsilon_sweep": sweep_summary_paper_aligned(FairLendConfig()),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="FairLend HMDA lending audit evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("demo")
    sub.add_parser("plan")
    sub.add_parser("audit", help="Run three-stage fair lending audit stub")
    sub.add_parser("binning", help="Standard vs fair income binning + ε sweep")
    args = p.parse_args()
    return {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "smoke": _cmd_smoke,
        "demo": _cmd_demo,
        "plan": _cmd_plan,
        "audit": _cmd_audit,
        "binning": _cmd_binning,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
