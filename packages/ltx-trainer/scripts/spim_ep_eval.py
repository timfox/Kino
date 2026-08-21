#!/usr/bin/env python3
"""SPIM Equilibrium Propagation CLI (arXiv:2606.13454)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.spim_ep.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.spim_ep.config import PAPER_URL, SPIMEPConfig  # noqa: E402
from ltx_trainer.spim_ep.paper import framework_card  # noqa: E402
from ltx_trainer.spim_ep.pipeline import evaluate_wine, evaluation_demo_run  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.spim_ep.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = SPIMEPConfig(rank=args.rank, n_free=args.n_free, n_nudge=args.n_nudge)
    out = evaluate_wine(cfg, seed=args.seed, train_epochs=args.epochs, max_train=args.max_train)
    print(json.dumps({"paper": PAPER_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="SPIM-EP hybrid optical EP (arXiv:2606.13454)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--epochs", type=int, default=2)
    demo.add_argument("--max-train", type=int, default=40)
    demo.add_argument("--rank", type=int, default=8)
    demo.add_argument("--n-free", type=int, default=5)
    demo.add_argument("--n-nudge", type=int, default=3)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
