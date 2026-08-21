"""CLI for SpecBench (arXiv:2605.30314)."""

from __future__ import annotations

import argparse
import json
import sys

from ltx_trainer.specbench.benchmark import benchmark_card, evaluate_agent, paper_agent_leaderboard
from ltx_trainer.specbench.mock import evaluation_smoke
from ltx_trainer.specbench.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    full_benchmark_demo,
    knowledge_card,
)
from ltx_trainer.specbench.scoring import score_task
from ltx_trainer.specbench.tasks import codex54_predictions_kep4671, task_by_id


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SpecBench RFC specification-deficiency utilities")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("framework")
    sub.add_parser("knowledge")
    sub.add_parser("benchmarks")
    sub.add_parser("smoke")
    sub.add_parser("leaderboard")
    d = sub.add_parser("demo")
    d.add_argument("--task", default="kubernetes_kep4671")
    sub.add_parser("full")

    args = p.parse_args(argv)
    if args.command == "framework":
        _print(framework_card())
    elif args.command == "knowledge":
        _print(knowledge_card())
    elif args.command == "benchmarks":
        _print(benchmarks_bundle())
    elif args.command == "smoke":
        _print(evaluation_smoke())
    elif args.command == "leaderboard":
        _print({"leaderboard": paper_agent_leaderboard()})
    elif args.command == "demo":
        _print(evaluation_demo(task_id=args.task))
    elif args.command == "full":
        _print(full_benchmark_demo())
    else:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
