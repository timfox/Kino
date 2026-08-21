#!/usr/bin/env python3
"""PrismAvatar CLI (arXiv:2606.10550)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.prism_avatar.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.prism_avatar.config import PAPER_URL  # noqa: E402
from ltx_trainer.prism_avatar.paper import framework_card  # noqa: E402
from ltx_trainer.prism_avatar.pipeline import (  # noqa: E402
    run_display_smoke,
    run_marcel_metrics_smoke,
    run_training_smoke,
)


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.prism_avatar.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out = {
        "paper": PAPER_URL,
        "train": run_training_smoke(seed=args.seed),
        "display": run_display_smoke(seed=args.seed),
        "marcel": run_marcel_metrics_smoke(seed=args.seed),
    }
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="PrismAvatar PMV avatar + lenticular display (arXiv:2606.10550)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
