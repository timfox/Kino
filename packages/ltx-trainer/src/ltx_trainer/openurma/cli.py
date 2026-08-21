"""CLI for OpenURMA (arXiv:2605.28717)."""

from __future__ import annotations

import argparse
import json
import sys

from ltx_trainer.openurma.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
)


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="OpenURMA — Unified Bus clean-room stub")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("framework")
    sub.add_parser("knowledge")
    sub.add_parser("benchmarks")
    d = sub.add_parser("demo")
    d.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("state")
    s.add_argument("--n", type=int, default=1024)
    s.add_argument("--m", type=int, default=None)

    args = p.parse_args(argv)
    if args.command == "framework":
        _print(framework_card())
    elif args.command == "knowledge":
        _print(knowledge_card())
    elif args.command == "benchmarks":
        _print(benchmarks_bundle())
    elif args.command == "demo":
        _print(evaluation_demo(seed=args.seed))
    elif args.command == "state":
        from ltx_trainer.openurma.state_model import state_at_scale

        m = args.m if args.m is not None else args.n
        _print(state_at_scale(args.n, m))
    else:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
