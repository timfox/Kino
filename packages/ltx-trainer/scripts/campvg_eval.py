#!/usr/bin/env python3
"""CamPVG CLI (Ji et al., arXiv:2509.19979)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.campvg.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.campvg.config import PAPER_URL, CamPVGConfig  # noqa: E402
from ltx_trainer.campvg.datasets import datasets_card  # noqa: E402
from ltx_trainer.campvg.paper import framework_card  # noqa: E402
from ltx_trainer.campvg.pipeline import (  # noqa: E402
    ablation_components,
    epipolar_k_demo,
    evaluation_demo_run,
    plucker_demo,
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
    from ltx_trainer.campvg.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_ablation(args: argparse.Namespace) -> int:
    from ltx_trainer.campvg.benchmarks import TABLE2_ABLATION

    cfg = CamPVGConfig(height=args.height, width=args.width, num_frames=args.frames, epipolar_k=8)
    print(
        json.dumps(
            {"table2_paper": TABLE2_ABLATION, "stub_forward": ablation_components(cfg)},
            indent=2,
        )
    )
    return 0


def _cmd_plucker(args: argparse.Namespace) -> int:
    cfg = CamPVGConfig(height=args.height, width=args.width)
    print(json.dumps({"paper": PAPER_URL, **plucker_demo(cfg)}, indent=2))
    return 0


def _cmd_epipolar_k(args: argparse.Namespace) -> int:
    from ltx_trainer.campvg.benchmarks import TABLE3_EPIPOLAR_K

    k = args.k
    print(
        json.dumps(
            {"table3_paper": TABLE3_EPIPOLAR_K, "stub": epipolar_k_demo(k)},
            indent=2,
        )
    )
    return 0


def _cmd_train_stub(args: argparse.Namespace) -> int:
    cfg = CamPVGConfig(
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        epipolar_k=min(args.k, 32),
    )
    print(json.dumps(train_step(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = CamPVGConfig(
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        epipolar_k=min(args.k, 32),
    )
    print(json.dumps({"paper": PAPER_URL, **evaluation_demo_run(cfg)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="CamPVG panoramic camera-controlled video")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    ab = sub.add_parser("ablation")
    ab.add_argument("--height", type=int, default=64)
    ab.add_argument("--width", type=int, default=128)
    ab.add_argument("--frames", type=int, default=4)
    ab.set_defaults(func=_cmd_ablation)

    plk = sub.add_parser("plucker")
    plk.add_argument("--height", type=int, default=64)
    plk.add_argument("--width", type=int, default=128)
    plk.set_defaults(func=_cmd_plucker)

    ek = sub.add_parser("epipolar-k")
    ek.add_argument("--k", type=int, default=250)
    ek.set_defaults(func=_cmd_epipolar_k)

    tr = sub.add_parser("train-stub")
    tr.add_argument("--height", type=int, default=64)
    tr.add_argument("--width", type=int, default=128)
    tr.add_argument("--frames", type=int, default=4)
    tr.add_argument("--k", type=int, default=16)
    tr.set_defaults(func=_cmd_train_stub)

    demo = sub.add_parser("demo")
    demo.add_argument("--height", type=int, default=64)
    demo.add_argument("--width", type=int, default=128)
    demo.add_argument("--frames", type=int, default=4)
    demo.add_argument("--k", type=int, default=16)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
