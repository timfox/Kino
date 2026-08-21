#!/usr/bin/env python3
"""SemanticStitch CLI (arXiv:2511.12084)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.semantic_stitch.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.semantic_stitch.config import PAPER_URL  # noqa: E402
from ltx_trainer.semantic_stitch.paper import framework_card  # noqa: E402
from ltx_trainer.semantic_stitch.pipeline import run_proceduralsky_bridge, run_stitch_smoke  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.semantic_stitch.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out = {"paper": PAPER_URL, "stitch": run_stitch_smoke(seed=args.seed)}
    print(json.dumps(out, indent=2))
    return 0


def _cmd_proceduralsky(args: argparse.Namespace) -> int:
    print(json.dumps(run_proceduralsky_bridge(skies_project=args.skies_project), indent=2))
    return 0


def _cmd_ltx_plan(args: argparse.Namespace) -> int:
    from ltx_trainer.semantic_stitch.ltx_plan import gopex_env_exports, ltx_training_plan

    print(
        json.dumps(
            {
                "plan": ltx_training_plan(skies_project=args.skies_project),
                "env": gopex_env_exports(args.skies_project),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="SemanticStitch (arXiv:2511.12084)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=_cmd_demo)
    ps = sub.add_parser("proceduralsky")
    ps.add_argument("--skies-project", default="sphere360-hdr-timelapse")
    ps.set_defaults(func=_cmd_proceduralsky)
    lp = sub.add_parser("ltx-plan")
    lp.add_argument("--skies-project", default="sphere360-hdr-timelapse")
    lp.set_defaults(func=_cmd_ltx_plan)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
