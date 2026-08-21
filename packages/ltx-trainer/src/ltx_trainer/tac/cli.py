"""CLI for Thinking as Compression (arXiv:2605.28713)."""

from __future__ import annotations

import argparse
import json
import sys

from ltx_trainer.tac.pipeline import benchmarks_bundle, evaluation_demo, framework_card, knowledge_card


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Thinking as Compression (TaC / TaC-C)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("framework")
    sub.add_parser("knowledge")
    sub.add_parser("benchmarks")
    d = sub.add_parser("demo")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--ratio", type=int, default=4, choices=(4, 8))

    b = sub.add_parser("batch")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--ratio", type=int, default=4, choices=(4, 8))

    args = p.parse_args(argv)
    if args.command == "framework":
        _print(framework_card())
    elif args.command == "knowledge":
        _print(knowledge_card())
    elif args.command == "benchmarks":
        _print(benchmarks_bundle())
    elif args.command == "demo":
        _print(evaluation_demo(seed=args.seed, compression_ratio=args.ratio))
    elif args.command == "batch":
        from ltx_trainer.tac.batch import evaluate_corpus
        from ltx_trainer.tac.corpus import builtin_corpus

        _print(evaluate_corpus(builtin_corpus(), compression_ratio=args.ratio, seed=args.seed))
    else:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
