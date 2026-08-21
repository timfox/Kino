#!/usr/bin/env python3
"""Physics-guided flood prediction CLI (Gebre et al., arXiv:2606.06524)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.flood_physics.benchmarks import hydro_metrics_card, summary_anchors, table_2_extent  # noqa: E402
from ltx_trainer.flood_physics.config import FloodPhysicsConfig  # noqa: E402
from ltx_trainer.flood_physics.losses import loss_components_card  # noqa: E402
from ltx_trainer.flood_physics.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.flood_physics.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.flood_physics.pipeline import run_demo  # noqa: E402
from ltx_trainer.flood_physics.swe import swe_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {"table_2": table_2_extent(), "hydro_metrics": hydro_metrics_card(), "summary": summary_anchors()},
            indent=2,
        )
    )
    return 0


def _cmd_method(_: argparse.Namespace) -> int:
    print(json.dumps({"swe": swe_card(), "losses": loss_components_card()}, indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = FloodPhysicsConfig(lambda_phys=args.lambda_phys, fno_modes=args.fno_modes)
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Physics-guided UNet+FNO flood prediction stub")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("paper", _cmd_paper),
        ("tables", _cmd_tables),
        ("method", _cmd_method),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    demo = sub.add_parser("demo")
    demo.add_argument("--lambda-phys", type=float, default=0.1)
    demo.add_argument("--fno-modes", type=int, default=16)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
