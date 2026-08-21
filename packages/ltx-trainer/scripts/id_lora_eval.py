#!/usr/bin/env python3
"""ID-LoRA knowledge, prompt, and inference-plan CLI."""

from __future__ import annotations

import argparse
import json
import sys


def _emit(obj: object) -> None:
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def main() -> int:
    from ltx_trainer.id_lora import (
        build_structured_prompt,
        evaluation_demo,
        evaluation_smoke,
        framework_card,
        knowledge_card,
        parse_id_lora_prompt,
        training_plan,
    )
    from ltx_trainer.id_lora.inference_plan import build_inference_argv, inference_defaults

    p = argparse.ArgumentParser(description="ID-LoRA integration CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("paper")
    sub.add_parser("framework")
    sub.add_parser("training")
    sub.add_parser("smoke")
    sub.add_parser("defaults")

    demo = sub.add_parser("demo")
    demo.add_argument("--speech", default="We are proud to introduce ID-LoRA.")
    demo.add_argument("--mode", default="one_stage", choices=["one_stage", "two_stage"])

    prompt = sub.add_parser("prompt")
    prompt.add_argument("--visual", required=True)
    prompt.add_argument("--speech", required=True)
    prompt.add_argument("--sounds", default="")

    parse = sub.add_parser("parse")
    parse.add_argument("text")

    plan = sub.add_parser("plan")
    plan.add_argument("--mode", default="one_stage", choices=["one_stage", "two_stage", "two_stage_hq"])
    plan.add_argument("--ltx-version", default="2", choices=["2", "2.3"])
    plan.add_argument("--prompt", default="")

    args = p.parse_args()
    if args.cmd == "knowledge":
        _emit(knowledge_card())
    elif args.cmd == "paper":
        _emit(framework_card())
    elif args.cmd == "framework":
        _emit(framework_card())
    elif args.cmd == "training":
        _emit(training_plan())
    elif args.cmd == "smoke":
        _emit(evaluation_smoke())
    elif args.cmd == "defaults":
        _emit(inference_defaults())
    elif args.cmd == "demo":
        _emit(evaluation_demo(speech=args.speech, mode=args.mode))
    elif args.cmd == "prompt":
        _emit({"prompt": build_structured_prompt(args.visual, args.speech, args.sounds)})
    elif args.cmd == "parse":
        parsed = parse_id_lora_prompt(args.text)
        _emit(
            {
                "visual": parsed.visual,
                "speech": parsed.speech,
                "sounds": parsed.sounds,
                "identity_ready": parsed.identity_ready,
            }
        )
    elif args.cmd == "plan":
        _emit(build_inference_argv(mode=args.mode, prompt=args.prompt or None, ltx_version=args.ltx_version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
