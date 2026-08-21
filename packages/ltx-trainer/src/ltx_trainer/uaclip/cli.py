"""CLI for Utility-Aware CLIP (arXiv:2605.28733)."""

from __future__ import annotations

import argparse
import json
import sys

from ltx_trainer.uaclip.pipeline import benchmarks_bundle, evaluation_demo, framework_card, knowledge_card


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Utility-Aware CLIP / Generator (Feng & Xie)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("framework")
    sub.add_parser("knowledge")
    sub.add_parser("benchmarks")
    d = sub.add_parser("demo")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--platform", choices=("amazon", "airbnb"), default="amazon")
    tr = sub.add_parser("train")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--platform", choices=("amazon", "airbnb"), default="amazon")

    args = p.parse_args(argv)
    if args.command == "framework":
        _print(framework_card())
    elif args.command == "knowledge":
        _print(knowledge_card())
    elif args.command == "benchmarks":
        _print(benchmarks_bundle())
    elif args.command == "demo":
        _print(evaluation_demo(seed=args.seed, platform=args.platform))
    elif args.command == "train":
        from ltx_trainer.uaclip.training import demo_train

        _print(demo_train(seed=args.seed, platform=args.platform))
    else:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
