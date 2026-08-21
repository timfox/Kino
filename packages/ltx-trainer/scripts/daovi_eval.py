#!/usr/bin/env python3
"""DAOVI CLI (Seshimo & Isogawa, arXiv:2509.00396)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.daovi.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.daovi.config import PAPER_URL, DaoviConfig  # noqa: E402
from ltx_trainer.daovi.datasets import datasets_card  # noqa: E402
from ltx_trainer.daovi.paper import framework_card  # noqa: E402
from ltx_trainer.daovi.pipeline import (  # noqa: E402
    ablation_modules,
    distortion_demo,
    evaluation_demo_run,
    geodesic_demo,
    train_step,
)


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(datasets_card(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.daovi.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_ablation(args: argparse.Namespace) -> int:
    from ltx_trainer.daovi.benchmarks import TABLE2_ABLATION

    cfg = DaoviConfig(height=args.height, width=args.width, num_frames=args.frames)
    print(
        json.dumps(
            {"table2_paper": TABLE2_ABLATION, "stub_forward": ablation_modules(cfg)},
            indent=2,
        )
    )
    return 0


def _cmd_geodesic(args: argparse.Namespace) -> int:
    print(json.dumps(geodesic_demo(args.width, args.height), indent=2))
    return 0


def _cmd_distortion(args: argparse.Namespace) -> int:
    print(json.dumps(distortion_demo(args.height, args.width), indent=2))
    return 0


def _cmd_train_stub(args: argparse.Namespace) -> int:
    cfg = DaoviConfig(height=args.height, width=args.width, num_frames=args.frames)
    print(json.dumps(train_step(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = DaoviConfig(height=args.height, width=args.width, num_frames=args.frames)
    print(json.dumps({"paper": PAPER_URL, **evaluation_demo_run(cfg)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="DAOVI omnidirectional video inpainting stub")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("tables", _cmd_tables),
        ("dataset", _cmd_dataset),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)
    for name, fn, extra in (
        ("demo", _cmd_demo, True),
        ("ablation", _cmd_ablation, True),
        ("geodesic", _cmd_geodesic, True),
        ("distortion", _cmd_distortion, True),
        ("train-stub", _cmd_train_stub, True),
    ):
        sp = sub.add_parser(name)
        if extra:
            sp.add_argument("--height", type=int, default=48)
            sp.add_argument("--width", type=int, default=96)
            sp.add_argument("--frames", type=int, default=4)
        sp.set_defaults(func=fn)
    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
