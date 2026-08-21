"""CLI entry for Qwen3-TTS inference (requires qwen-tts + GPU for real audio)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ltx_trainer.qwen3_tts.dialogue import RoleBank
from ltx_trainer.qwen3_tts.runtime import GenerationParams, Qwen3TtsRuntime, write_wav


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Qwen3-TTS inference")
    p.add_argument("--mode", required=True, choices=[
        "custom_voice", "voice_design", "voice_clone", "voice_clone_prompt", "dialogue",
    ])
    p.add_argument("--text", default="Hello from Qwen3-TTS.")
    p.add_argument("--output", default="output.wav")
    p.add_argument("--model", default=None)
    p.add_argument("--model-choice", default="1.7B", choices=["0.6B", "1.7B"])
    p.add_argument("--language", default="Auto")
    p.add_argument("--speaker", default="Ryan")
    p.add_argument("--instruct", default=None)
    p.add_argument("--ref-audio", default=None)
    p.add_argument("--ref-text", default=None)
    p.add_argument("--attention", default="auto")
    p.add_argument("--unload", action="store_true")
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--script", default=None, help="Dialogue script file or inline Role: text")
    p.add_argument("--role-bank", default=None, help="JSON map role -> prompt pickle path")
    p.add_argument("--prompt-out", default=None, help="Write voice clone prompt pickle")
    p.add_argument("--merge-dialogue", action="store_true", default=True)
    p.add_argument("--pause-seconds", type=float, default=None)
    args = p.parse_args(argv)

    rt = Qwen3TtsRuntime()
    gen = GenerationParams(
        top_p=args.top_p if args.top_p is not None else rt.cfg.top_p,
        top_k=args.top_k if args.top_k is not None else rt.cfg.top_k,
        temperature=args.temperature if args.temperature is not None else rt.cfg.temperature,
        repetition_penalty=(
            args.repetition_penalty
            if args.repetition_penalty is not None
            else rt.cfg.repetition_penalty
        ),
    )

    try:
        if args.mode == "custom_voice":
            wavs, sr = rt.generate_custom_voice(
                text=args.text,
                speaker=args.speaker,
                language=args.language,
                instruct=args.instruct,
                model_choice=args.model_choice,  # type: ignore[arg-type]
                attention=args.attention,
                params=gen,
                unload=args.unload,
            )
            write_wav(args.output, wavs[0], sr)

        elif args.mode == "voice_design":
            if not args.instruct:
                print("voice_design requires --instruct", file=sys.stderr)
                return 2
            wavs, sr = rt.generate_voice_design(
                text=args.text,
                instruct=args.instruct,
                language=args.language,
                attention=args.attention,
                params=gen,
                unload=args.unload,
            )
            write_wav(args.output, wavs[0], sr)

        elif args.mode == "voice_clone_prompt":
            if not args.ref_audio:
                print("voice_clone_prompt requires --ref-audio", file=sys.stderr)
                return 2
            prompt = rt.create_voice_clone_prompt(
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                model_choice=args.model_choice,  # type: ignore[arg-type]
                attention=args.attention,
                unload=args.unload,
            )
            out = args.prompt_out or str(Path(args.output).with_suffix(".prompt.pkl"))
            import pickle  # noqa: PLC0415

            Path(out).write_bytes(pickle.dumps(prompt))
            print(json.dumps({"prompt_out": out}))

        elif args.mode == "voice_clone":
            wavs, sr = rt.generate_voice_clone(
                text=args.text,
                language=args.language,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                model_choice=args.model_choice,  # type: ignore[arg-type]
                attention=args.attention,
                params=gen,
                unload=args.unload,
            )
            write_wav(args.output, wavs[0], sr)

        elif args.mode == "dialogue":
            script = args.script or args.text
            if args.role_bank:
                bank_data = json.loads(Path(args.role_bank).read_text(encoding="utf-8"))
                import pickle  # noqa: PLC0415

                bank = RoleBank()
                for role, pkl_path in bank_data.items():
                    with Path(pkl_path).open("rb") as fh:
                        bank.add(role, pickle.load(fh))  # noqa: S301
            else:
                print("dialogue requires --role-bank JSON", file=sys.stderr)
                return 2
            wavs, sr, lines = rt.generate_dialogue(
                script=script,
                role_bank=bank,
                model_choice=args.model_choice,  # type: ignore[arg-type]
                attention=args.attention,
                pause_seconds=args.pause_seconds,
                merge_outputs=args.merge_dialogue,
                language=args.language,
                params=gen,
                unload=args.unload,
            )
            write_wav(args.output, wavs[0], sr)
            print(json.dumps({"lines": len(lines), "sample_rate": sr}))

    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, "output": args.output, "mode": args.mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
