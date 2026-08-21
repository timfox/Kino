#!/usr/bin/env python3
"""Clairvoyant predictive SJF sidecar CLI (Sundaresan, arXiv:2606.07248)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.clairvoyant.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.clairvoyant.datasets import datasets_card  # noqa: E402
from ltx_trainer.clairvoyant.features import extract_features  # noqa: E402
from ltx_trainer.clairvoyant.paper import evaluation_demo, framework_card, knowledge_blob  # noqa: E402
from ltx_trainer.clairvoyant.pipeline import evaluation_demo_run  # noqa: E402
from ltx_trainer.clairvoyant.predictor import predict_record  # noqa: E402
from ltx_trainer.clairvoyant.queueing import discrete_event_simulation, tau_sensitivity_table  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_blob(), indent=2))
    return 0


def _cmd_framework(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_datasets(_: argparse.Namespace) -> int:
    print(json.dumps(datasets_card(), indent=2))
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    feats = extract_features(args.prompt)
    out = {k: v for k, v in feats.items() if k != "vector"}
    print(json.dumps(out, indent=2))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    print(json.dumps(predict_record(args.prompt, variant=args.variant), indent=2))
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    if args.tau_table:
        print(json.dumps(tau_sensitivity_table(), indent=2))
        return 0
    out = discrete_event_simulation(
        n_requests=args.n,
        policy=args.policy,
        starvation_tau_s=args.tau,
        seed=args.seed,
    )
    print(json.dumps(out, indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo_run(variant=args.variant), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.clairvoyant.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_full_demo(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Clairvoyant SJF sidecar stub CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("framework").set_defaults(func=_cmd_framework)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("datasets").set_defaults(func=_cmd_datasets)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    sub.add_parser("full-demo").set_defaults(func=_cmd_full_demo)

    pf = sub.add_parser("features")
    pf.add_argument("prompt")
    pf.set_defaults(func=_cmd_features)

    pp = sub.add_parser("predict")
    pp.add_argument("prompt")
    pp.add_argument("--variant", default="sharegpt", choices=("sharegpt", "lmsys", "oasst1"))
    pp.set_defaults(func=_cmd_predict)

    ps = sub.add_parser("simulate")
    ps.add_argument("--policy", default="sjf", choices=("sjf", "fcfs"))
    ps.add_argument("--n", type=int, default=2000)
    ps.add_argument("--tau", type=float, default=10.5)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--tau-table", action="store_true")
    ps.set_defaults(func=_cmd_simulate)

    pd = sub.add_parser("demo")
    pd.add_argument("--variant", default="sharegpt", choices=("sharegpt", "lmsys", "oasst1"))
    pd.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
