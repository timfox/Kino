#!/usr/bin/env python3
"""dots.tts knowledge, upstream status, and inference-plan CLI."""

from __future__ import annotations

import argparse
import json
import sys


def _emit(obj: object) -> None:
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def main() -> int:
    from ltx_trainer.dotstts import (
        benchmarks_bundle,
        evaluation_demo,
        evaluation_smoke,
        framework_card,
        headline_results,
        pipeline_demo,
    )
    from ltx_trainer.dotstts.upstream import (
        build_gradio_argv,
        build_infer_argv,
        build_prepare_data_argv,
        build_train_argv,
        hf_checkpoints,
        install_plan,
        upstream_knowledge,
        upstream_status,
    )

    p = argparse.ArgumentParser(description="dots.tts integration CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("paper")
    sub.add_parser("framework")
    sub.add_parser("smoke")
    sub.add_parser("status")
    sub.add_parser("install")
    sub.add_parser("checkpoints")

    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=42)

    plan = sub.add_parser("plan")
    plan.add_argument("--text", default="Hello, this is dots.tts.")
    plan.add_argument("--output", default="output.wav")
    plan.add_argument("--prompt-audio", default=None)
    plan.add_argument("--prompt-text", default=None)
    plan.add_argument(
        "--checkpoint",
        default="soar",
        choices=["base", "soar", "mf"],
    )
    plan.add_argument("--model-name-or-path", default=None)
    plan.add_argument("--num-steps", type=int, default=None)

    gradio = sub.add_parser("gradio-plan")
    gradio.add_argument("--checkpoint", default="soar", choices=["base", "soar", "mf"])
    gradio.add_argument("--port", type=int, default=7860)

    sub.add_parser("prepare-plan")
    sub.add_parser("train-plan")

    args = p.parse_args()
    if args.cmd == "knowledge":
        _emit(upstream_knowledge())
    elif args.cmd in {"paper", "framework"}:
        _emit(framework_card())
    elif args.cmd == "smoke":
        _emit(evaluation_smoke())
    elif args.cmd == "status":
        _emit(upstream_status())
    elif args.cmd == "install":
        _emit(install_plan())
    elif args.cmd == "checkpoints":
        _emit(hf_checkpoints())
    elif args.cmd == "demo":
        _emit(
            {
                "evaluation": evaluation_demo(seed=args.seed),
                "pipeline": pipeline_demo(seed=args.seed),
                "headline": headline_results(),
                "benchmarks": benchmarks_bundle(),
            }
        )
    elif args.cmd == "plan":
        _emit(
            build_infer_argv(
                text=args.text,
                output=args.output,
                prompt_audio=args.prompt_audio,
                prompt_text=args.prompt_text,
                checkpoint=args.checkpoint,
                model_name_or_path=args.model_name_or_path,
                num_steps=args.num_steps,
            )
        )
    elif args.cmd == "gradio-plan":
        _emit(build_gradio_argv(checkpoint=args.checkpoint, port=args.port))
    elif args.cmd == "prepare-plan":
        _emit(build_prepare_data_argv())
    elif args.cmd == "train-plan":
        _emit(build_train_argv())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
