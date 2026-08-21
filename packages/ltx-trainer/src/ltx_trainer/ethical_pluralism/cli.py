"""CLI for ethical pluralism demos (arXiv:2605.28707)."""

from __future__ import annotations

import argparse
import json
import sys

from ltx_trainer.ethical_pluralism.benchmark import benchmark_card, generate_benchmark, train_test_split
from ltx_trainer.ethical_pluralism.config import EthicalPluralismConfig
from ltx_trainer.ethical_pluralism.ensemble import ablation_study
from ltx_trainer.ethical_pluralism.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
)
from ltx_trainer.ethical_pluralism.simplex import plurality_features, project_simplex


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ethical pluralism normative-simplex utilities")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("framework")
    sub.add_parser("knowledge")
    sub.add_parser("benchmarks")
    d = sub.add_parser("demo")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument(
        "--full",
        action="store_true",
        help="Production stacked ensemble on mini benchmark (no lexicon shortcut)",
    )
    ab = sub.add_parser("ablation")
    ab.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("simplex")
    s.add_argument("--alpha", type=float, default=0.4)
    s.add_argument("--beta", type=float, default=0.35)
    s.add_argument("--gamma", type=float, default=0.25)

    args = p.parse_args(argv)
    if args.command == "framework":
        _print(framework_card())
    elif args.command == "knowledge":
        _print(knowledge_card())
    elif args.command == "benchmarks":
        _print(benchmarks_bundle())
    elif args.command == "demo":
        _print(evaluation_demo(seed=args.seed, full=getattr(args, "full", False)))
    elif args.command == "ablation":
        cfg = EthicalPluralismConfig.production()
        cfg.random_seed = args.seed
        cases = generate_benchmark(
            seed=args.seed,
            cases_per_subtheory=cfg.mini_cases_per_subtheory,
            include_bracket_markers=False,
        )
        train, test = train_test_split(cases, cfg.train_fraction, seed=args.seed)
        _print({"ablation": ablation_study(train, test, cfg), "n_cases": len(cases)})
    elif args.command == "simplex":
        scores = project_simplex(args.alpha, args.beta, args.gamma)
        _print({"scores": {"alpha": scores[0], "beta": scores[1], "gamma": scores[2]}, **plurality_features(scores)})
    else:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
