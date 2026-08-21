"""CLI for AutoCut (arXiv:2603.28366)."""

from __future__ import annotations

import argparse
import json

from ltx_trainer.autocut.config import AutoCutConfig
from ltx_trainer.autocut.dataset_build import dataset_pipeline_summary
from ltx_trainer.autocut.mock import evaluation_smoke
from ltx_trainer.autocut.pipeline import (
    benchmarks_bundle,
    editing_demo,
    evaluation_demo,
    framework_card,
    knowledge_card,
)
from ltx_trainer.autocut.training import full_training_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="AutoCut: end-to-end ad video editing via multimodal discretization (arXiv:2603.28366)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("framework", help="Print framework card JSON")
    sub.add_parser("knowledge", help="Print knowledge card JSON")
    sub.add_parser("tables", help="Print paper tables bundle")
    sub.add_parser("smoke", help="Run evaluation smoke")
    sub.add_parser("train-config", help="Print RQ-VAE + alignment + SFT training configs")
    sub.add_parser("dataset", help="Print dataset construction pipeline summary")

    demo = sub.add_parser("demo", help="Run evaluation demo for a benchmark case")
    demo.add_argument("--case-id", default="edifier_earbuds")

    edit = sub.add_parser("edit", help="Run script-driven or footage-driven editing plan")
    edit.add_argument("--case-id", default="edifier_earbuds")
    edit.add_argument(
        "--scenario",
        choices=("script_driven", "footage_driven"),
        default="script_driven",
    )
    edit.add_argument(
        "--render-strategy",
        choices=("by_clip", "by_frame"),
        default="by_clip",
    )

    args = p.parse_args(argv)
    cfg = AutoCutConfig()

    if args.cmd == "framework":
        print(json.dumps(framework_card(cfg), indent=2))
    elif args.cmd == "knowledge":
        print(json.dumps(knowledge_card(cfg), indent=2))
    elif args.cmd == "tables":
        print(json.dumps(benchmarks_bundle(), indent=2))
    elif args.cmd == "smoke":
        print(json.dumps(evaluation_smoke(cfg), indent=2))
    elif args.cmd == "train-config":
        print(json.dumps(full_training_pipeline(cfg), indent=2))
    elif args.cmd == "dataset":
        print(json.dumps(dataset_pipeline_summary(cfg), indent=2))
    elif args.cmd == "demo":
        print(json.dumps(evaluation_demo(case_id=args.case_id, cfg=cfg), indent=2))
    elif args.cmd == "edit":
        from ltx_trainer.autocut.taxonomy import RenderStrategy

        strat = RenderStrategy.BY_CLIP if args.render_strategy == "by_clip" else RenderStrategy.BY_FRAME
        if args.scenario == "footage_driven":
            from ltx_trainer.autocut.inference import run_footage_driven_edit

            out = run_footage_driven_edit(args.case_id, cfg=cfg)
        else:
            from ltx_trainer.autocut.inference import run_script_driven_edit

            out = run_script_driven_edit(args.case_id, cfg=cfg, render_strategy=strat)
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
