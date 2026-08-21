#!/usr/bin/env python3
"""Branch-level energy localization CLI (Montoya et al., arXiv:2606.07076)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.branch_energy.benchmarks import summary_anchors, table_i_rows  # noqa: E402
from ltx_trainer.branch_energy.branch import branch_card  # noqa: E402
from ltx_trainer.branch_energy.config import BranchEnergyConfig  # noqa: E402
from ltx_trainer.branch_energy.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.branch_energy.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.branch_energy.pipeline import run_demo  # noqa: E402
from ltx_trainer.branch_energy.theorems import theorems_card  # noqa: E402
from ltx_trainer.branch_energy.topology import topology_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps({"table_i": table_i_rows(), "summary": summary_anchors()}, indent=2))
    return 0


def _cmd_method(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {"theorems": theorems_card(), "branch_balance": branch_card(), "topology": topology_card()},
            indent=2,
        )
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = BranchEnergyConfig(f0_hz=args.f0, fs_hz=args.fs, num_periods=args.periods)
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_classical(_: argparse.Namespace) -> int:
    from ltx_trainer.branch_energy.classical import balanced_rl_phase_comparison, classical_card, classical_comparisons

    print(
        json.dumps(
            {"classical": classical_card(), "comparisons": classical_comparisons()},
            indent=2,
        )
    )
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Branch-level three-phase energy localization stub")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("paper", _cmd_paper),
        ("tables", _cmd_tables),
        ("method", _cmd_method),
        ("classical", _cmd_classical),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    demo = sub.add_parser("demo")
    demo.add_argument("--f0", type=float, default=50.0)
    demo.add_argument("--fs", type=float, default=10_000.0)
    demo.add_argument("--periods", type=float, default=1.0)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
