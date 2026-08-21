"""VarRate CLI (arXiv:2607.15498)."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ltx_trainer.varrate")
    p.add_argument("command", choices=("framework", "knowledge", "benchmarks", "demo", "smoke"))
    p.add_argument("--n-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.command == "framework":
        from ltx_trainer.varrate import VarRateConfig, framework_card

        print(json.dumps({"framework": framework_card(), "config": VarRateConfig().__dict__}, indent=2))
    elif args.command == "knowledge":
        from ltx_trainer.varrate import knowledge_card

        print(json.dumps(knowledge_card(), indent=2))
    elif args.command == "benchmarks":
        from ltx_trainer.varrate import benchmarks_bundle

        print(json.dumps(benchmarks_bundle(), indent=2))
    elif args.command == "demo":
        from ltx_trainer.varrate import evaluation_demo

        print(json.dumps(evaluation_demo(n_tokens=args.n_tokens, seed=args.seed), indent=2))
    else:
        from ltx_trainer.varrate import evaluation_smoke

        out = evaluation_smoke()
        print(json.dumps(out, indent=2))
        return 0 if out.get("all_pass") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
