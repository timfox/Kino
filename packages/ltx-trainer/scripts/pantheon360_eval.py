#!/usr/bin/env python3
"""Pantheon360 CLI (Chen et al., arXiv:2605.25449)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.pantheon360.config import Pantheon360Config  # noqa: E402
from ltx_trainer.pantheon360.integration import (  # noqa: E402
    proceduralsky_card,
    proceduralsky_pipeline_plan,
)
from ltx_trainer.pantheon360.ltx_plan import gopex_env_exports, ltx_training_plan  # noqa: E402
from ltx_trainer.pantheon360.pipeline import (  # noqa: E402
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    paper_checks,
    training_step_demo,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Pantheon360 3D-aware 360° video diffusion stub")
    p.add_argument(
        "command",
        choices=(
            "knowledge",
            "tables",
            "smoke",
            "demo",
            "checks",
            "proceduralsky",
            "ltx-plan",
            "env",
        ),
    )
    p.add_argument("--skies-project", default="sphere360-hdr-timelapse")
    args = p.parse_args()
    cfg = Pantheon360Config()
    print(f"# {cfg.project_page}", file=sys.stderr)

    if args.command == "knowledge":
        print(json.dumps(framework_card(), indent=2))
    elif args.command == "tables":
        print(json.dumps(benchmarks_bundle(), indent=2))
    elif args.command == "smoke":
        from ltx_trainer.pantheon360.mock import evaluation_smoke

        print(json.dumps(evaluation_smoke(), indent=2))
    elif args.command == "demo":
        print(json.dumps({"training_step": training_step_demo(), "evaluation": evaluation_demo()}, indent=2))
    elif args.command == "checks":
        print(json.dumps(paper_checks(), indent=2))
    elif args.command == "proceduralsky":
        print(
            json.dumps(
                {
                    "card": proceduralsky_card(),
                    "pipeline": proceduralsky_pipeline_plan(skies_project=args.skies_project),
                },
                indent=2,
            )
        )
    elif args.command == "ltx-plan":
        print(json.dumps(ltx_training_plan(skies_project=args.skies_project), indent=2))
    else:
        print(json.dumps(gopex_env_exports(args.skies_project), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
