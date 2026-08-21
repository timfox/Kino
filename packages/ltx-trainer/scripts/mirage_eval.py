#!/usr/bin/env python3
"""Mirage CLI — latent spatial memory for video world models (arXiv:2606.09828)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.mirage.config import MirageConfig  # noqa: E402
from ltx_trainer.mirage.ltx_plan import ltx_training_plan  # noqa: E402
from ltx_trainer.mirage.pipeline import (  # noqa: E402
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
    paper_checks,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Mirage latent spatial memory stub")
    p.add_argument(
        "command",
        choices=("knowledge", "tables", "smoke", "demo", "checks", "ltx-plan", "rollout", "efficiency", "ablation", "algorithm", "env"),
    )
    p.add_argument("--n-chunks", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = MirageConfig()
    print(f"# {cfg.website}", file=sys.stderr)

    if args.command == "knowledge":
        print(json.dumps(knowledge_card(cfg), indent=2))
    elif args.command == "tables":
        print(json.dumps(benchmarks_bundle(), indent=2))
    elif args.command == "smoke":
        from ltx_trainer.mirage.mock import evaluation_smoke

        print(json.dumps(evaluation_smoke(cfg), indent=2))
    elif args.command == "demo":
        print(json.dumps(evaluation_demo(cfg), indent=2))
    elif args.command == "checks":
        print(json.dumps(paper_checks(), indent=2))
    elif args.command == "ltx-plan":
        print(json.dumps(ltx_training_plan(cfg), indent=2))
    elif args.command == "rollout":
        from ltx_trainer.mirage.rollout import run_toy_rollout

        print(json.dumps(run_toy_rollout(cfg, n_chunks=args.n_chunks, seed=args.seed), indent=2))
    elif args.command == "efficiency":
        from ltx_trainer.mirage.efficiency import efficiency_demo

        print(json.dumps(efficiency_demo(cfg), indent=2))
    elif args.command == "ablation":
        from ltx_trainer.mirage.ablation import run_ablation_smoke

        print(json.dumps(run_ablation_smoke(cfg), indent=2))
    elif args.command == "algorithm":
        from ltx_trainer.mirage.algorithm import algorithm1_steps, run_algorithm1_stub

        print(
            json.dumps(
                {
                    "steps": algorithm1_steps(cfg),
                    "stub": run_algorithm1_stub(cfg, n_chunks=args.n_chunks, seed=args.seed),
                },
                indent=2,
            )
        )
    elif args.command == "env":
        from ltx_trainer.mirage.ltx_plan import gopex_env_exports

        for k, v in gopex_env_exports().items():
            print(f'export {k}="{v}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
