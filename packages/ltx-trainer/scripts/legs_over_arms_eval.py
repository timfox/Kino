#!/usr/bin/env python3
"""Legs Over Arms / HST CLI (Le et al. arXiv:2602.09076)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.legs_over_arms.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.legs_over_arms.config import PAPER_URL, LegsOverArmsConfig  # noqa: E402
from ltx_trainer.legs_over_arms.datasets import datasets_bundle  # noqa: E402
from ltx_trainer.legs_over_arms.paper import framework_card  # noqa: E402
from ltx_trainer.legs_over_arms.pipeline import evaluation_demo_run  # noqa: E402
from ltx_trainer.legs_over_arms.skeleton import FeatureConfig  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_datasets(_: argparse.Namespace) -> int:
    print(json.dumps(datasets_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.legs_over_arms.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = LegsOverArmsConfig()
    feat: FeatureConfig = args.feature  # type: ignore[assignment]
    out = evaluation_demo_run(cfg, device=args.device, feature_config=feat)
    print(json.dumps({"paper": PAPER_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Legs Over Arms HST pose trajectory (arXiv:2602.09076)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("datasets").set_defaults(func=_cmd_datasets)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument(
        "--feature",
        default="K3D_L",
        choices=[
            "baseline",
            "K3D",
            "K3D_L",
            "K2D",
            "K2D_L",
        ],
    )
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
