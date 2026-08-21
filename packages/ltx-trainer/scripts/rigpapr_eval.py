#!/usr/bin/env python3
"""RigPAPR fixed-viewpoint PAPR rig animation CLI (Peng et al. arXiv:2606.06685)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.rigpapr.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.rigpapr.config import PAPER_URL, RigPAPRConfig  # noqa: E402
from ltx_trainer.rigpapr.integration import integration_bundle  # noqa: E402
from ltx_trainer.rigpapr.paper import evaluation_demo, framework_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.rigpapr.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    print(json.dumps(integration_bundle(), indent=2))
    return 0


def _cmd_lbs(args: argparse.Namespace) -> int:
    from ltx_trainer.rigpapr.lbs import lbs_demo

    print(json.dumps(lbs_demo(seed=args.seed), indent=2))
    return 0


def _cmd_papr(args: argparse.Namespace) -> int:
    from ltx_trainer.rigpapr.papr import papr_render_demo

    print(json.dumps(papr_render_demo(seed=args.seed), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = RigPAPRConfig(num_points=args.points, num_bones=args.bones)
    out = evaluation_demo(seed=args.seed)
    out["config"] = cfg.__dict__
    out["paper"] = PAPER_URL
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="RigPAPR evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("plan")

    lbs = sub.add_parser("lbs")
    lbs.add_argument("--seed", type=int, default=0)

    papr = sub.add_parser("papr")
    papr.add_argument("--seed", type=int, default=0)

    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--points", type=int, default=256)
    demo.add_argument("--bones", type=int, default=12)

    args = p.parse_args()
    return {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "smoke": _cmd_smoke,
        "plan": _cmd_plan,
        "lbs": _cmd_lbs,
        "papr": _cmd_papr,
        "demo": _cmd_demo,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
