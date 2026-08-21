#!/usr/bin/env python3
"""GSpaRC CLI (Nukapotula et al., arXiv:2511.22793)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.gsparc.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.gsparc.config import PAPER_URL, GSpaRCConfig  # noqa: E402
from ltx_trainer.gsparc.datasets import datasets_card  # noqa: E402
from ltx_trainer.gsparc.paper import framework_card  # noqa: E402
from ltx_trainer.gsparc.pipeline import evaluation_demo_run, train_step  # noqa: E402


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
    from ltx_trainer.gsparc.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_confidence(_: argparse.Namespace) -> int:
    from ltx_trainer.gsparc.benchmarks import ARGOS_CONFIDENCE

    print(json.dumps(ARGOS_CONFIDENCE, indent=2))
    return 0


def _cmd_train_stub(args: argparse.Namespace) -> int:
    cfg = GSpaRCConfig(
        spectrum_height=args.height,
        spectrum_width=args.width,
        num_gaussians=args.gaussians,
    )
    print(json.dumps(train_step(cfg), indent=2))
    return 0


def _cmd_hyperparams(_: argparse.Namespace) -> int:
    from ltx_trainer.gsparc.hyperparams import training_hyperparameters

    print(json.dumps(training_hyperparameters(), indent=2))
    return 0


def _cmd_downstream(_: argparse.Namespace) -> int:
    from ltx_trainer.gsparc.downstream import downstream_card

    print(json.dumps(downstream_card(), indent=2))
    return 0


def _cmd_ablation(args: argparse.Namespace) -> int:
    from ltx_trainer.gsparc.pipeline import ablation_distance

    cfg = GSpaRCConfig(
        spectrum_height=args.height,
        spectrum_width=args.width,
        num_gaussians=args.gaussians,
    )
    print(json.dumps(ablation_distance(cfg), indent=2))
    return 0


def _cmd_densify(args: argparse.Namespace) -> int:
    from ltx_trainer.gsparc.pipeline import densify_report

    cfg = GSpaRCConfig(num_gaussians=args.gaussians)
    print(json.dumps(densify_report(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = GSpaRCConfig(
        spectrum_height=args.height,
        spectrum_width=args.width,
        num_gaussians=args.gaussians,
        use_distance_attenuation=not args.no_distance,
    )
    print(json.dumps({"paper": PAPER_URL, **evaluation_demo_run(cfg)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="GSpaRC RF Gaussian splatting")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    sub.add_parser("confidence").set_defaults(func=_cmd_confidence)
    sub.add_parser("hyperparams").set_defaults(func=_cmd_hyperparams)
    sub.add_parser("downstream").set_defaults(func=_cmd_downstream)
    ab = sub.add_parser("ablation")
    ab.add_argument("--height", type=int, default=24)
    ab.add_argument("--width", type=int, default=48)
    ab.add_argument("--gaussians", type=int, default=16)
    ab.set_defaults(func=_cmd_ablation)
    dn = sub.add_parser("densify")
    dn.add_argument("--gaussians", type=int, default=32)
    dn.set_defaults(func=_cmd_densify)
    tr = sub.add_parser("train-stub")
    tr.add_argument("--height", type=int, default=24)
    tr.add_argument("--width", type=int, default=48)
    tr.add_argument("--gaussians", type=int, default=16)
    tr.set_defaults(func=_cmd_train_stub)
    demo = sub.add_parser("demo")
    demo.add_argument("--height", type=int, default=32)
    demo.add_argument("--width", type=int, default=64)
    demo.add_argument("--gaussians", type=int, default=32)
    demo.add_argument("--no-distance", action="store_true")
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
