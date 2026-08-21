#!/usr/bin/env python3
"""Neural UV Atlas CLI (Salehi arXiv:2606.10050)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.neural_uv.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.neural_uv.config import PAPER_URL, NeuralUVConfig  # noqa: E402
from ltx_trainer.neural_uv.integration import integration_bundle  # noqa: E402
from ltx_trainer.neural_uv.paper import evaluation_demo, framework_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.neural_uv.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    print(json.dumps(integration_bundle(), indent=2))
    return 0


def _cmd_siren(args: argparse.Namespace) -> int:
    from ltx_trainer.neural_uv.siren import siren_demo

    print(json.dumps(siren_demo(seed=args.seed), indent=2))
    return 0


def _cmd_solver(args: argparse.Namespace) -> int:
    from ltx_trainer.neural_uv.solver import solver_demo

    print(json.dumps(solver_demo(seed=args.seed), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = NeuralUVConfig(spectral_rank=args.k, omega0=args.omega0)
    out = evaluation_demo(seed=args.seed)
    out["config"] = cfg.__dict__
    out["paper"] = PAPER_URL
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Neural UV Atlas evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("plan")

    siren = sub.add_parser("siren")
    siren.add_argument("--seed", type=int, default=0)

    solver = sub.add_parser("solver")
    solver.add_argument("--seed", type=int, default=0)

    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--k", type=int, default=16)
    demo.add_argument("--omega0", type=float, default=15.0)

    args = p.parse_args()
    return {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "smoke": _cmd_smoke,
        "plan": _cmd_plan,
        "siren": _cmd_siren,
        "solver": _cmd_solver,
        "demo": _cmd_demo,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
