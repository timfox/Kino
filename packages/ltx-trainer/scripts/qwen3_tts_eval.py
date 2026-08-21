#!/usr/bin/env python3
"""Qwen3-TTS knowledge, doctor, and inference-plan CLI."""

from __future__ import annotations

import argparse
import json
import sys


def _emit(obj: object) -> None:
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def main() -> int:
    from ltx_trainer.qwen3_tts import (
        benchmarks_bundle,
        evaluation_demo,
        evaluation_smoke,
        framework_card,
        headline_results,
        pipeline_demo,
    )
    from ltx_trainer.qwen3_tts.upstream import (
        build_infer_argv,
        doctor,
        install_plan,
        package_status,
        upstream_knowledge,
    )

    p = argparse.ArgumentParser(description="Qwen3-TTS integration CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("framework")
    sub.add_parser("smoke")
    sub.add_parser("status")
    sub.add_parser("doctor")
    sub.add_parser("install")

    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=42)

    plan = sub.add_parser("plan")
    plan.add_argument("--mode", default="custom_voice")
    plan.add_argument("--text", default="Hello from Qwen3-TTS.")
    plan.add_argument("--output", default="output.wav")
    plan.add_argument("--speaker", default="Ryan")
    plan.add_argument("--instruct", default=None)
    plan.add_argument("--ref-audio", default=None)
    plan.add_argument("--ref-text", default=None)
    plan.add_argument("--model-choice", default="1.7B")
    plan.add_argument("--attention", default="auto")

    args = p.parse_args()
    if args.cmd == "knowledge":
        _emit(upstream_knowledge())
    elif args.cmd == "framework":
        _emit(framework_card())
    elif args.cmd == "smoke":
        _emit(evaluation_smoke())
    elif args.cmd == "status":
        _emit(package_status())
    elif args.cmd == "doctor":
        _emit(doctor())
    elif args.cmd == "install":
        _emit(install_plan())
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
                mode=args.mode,  # type: ignore[arg-type]
                text=args.text,
                output=args.output,
                speaker=args.speaker,
                instruct=args.instruct,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                model_choice=args.model_choice,  # type: ignore[arg-type]
                attention=args.attention,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
