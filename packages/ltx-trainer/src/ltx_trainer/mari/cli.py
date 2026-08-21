"""CLI for MARI (arXiv:2605.28722)."""

from __future__ import annotations

import argparse
import json
import sys

from ltx_trainer.mari.inference import MARIPipeline
from ltx_trainer.mari.pipeline import benchmarks_bundle, evaluation_demo, framework_card, knowledge_card


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MARI — Multi-Adapter Representation Interventions")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("framework")
    sub.add_parser("knowledge")
    sub.add_parser("benchmarks")
    d = sub.add_parser("demo")
    d.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("forward")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--dim", type=int, default=64)

    args = p.parse_args(argv)
    if args.command == "framework":
        _print(framework_card())
    elif args.command == "knowledge":
        _print(knowledge_card())
    elif args.command == "benchmarks":
        _print(benchmarks_bundle())
    elif args.command == "demo":
        _print(evaluation_demo(seed=args.seed))
    elif args.command == "train":
        from ltx_trainer.mari.training import demo_train

        _print(demo_train(seed=args.seed))
    elif args.command == "forward":
        import numpy as np

        from ltx_trainer.mari.config import MARIConfig

        cfg = MARIConfig(hidden_dim=args.dim, num_adapters=3)
        pipe = MARIPipeline.from_config(cfg, seed=args.seed)
        h = np.random.default_rng(args.seed).standard_normal(args.dim)
        _print(pipe.forward_mc(h).to_dict())
    else:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
