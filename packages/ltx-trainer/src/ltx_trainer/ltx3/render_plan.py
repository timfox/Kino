"""Turn minute compose plans into executable inference + ffmpeg steps."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltx_trainer.ltx3.infer_env import InferEnv
from ltx_trainer.ltx3.minute_compose import MinuteComposePlan, SegmentSpec, compose_minute_plan, segment_from_events
from ltx_trainer.ltx3.temporal_upscale import temporal_upscale_post_concat_command
from ltx_trainer.longav_compass.annotation import BenchmarkCase, example_t2av_performance_ads_l4


@dataclass
class SegmentRenderStep:
    index: int
    output_mp4: str
    hero_in: str | None
    hero_out: str | None
    infer_shell: str
    extract_hero_shell: str | None
    frames: int
    prompt_preview: str
    negative_prompt_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "output_mp4": self.output_mp4,
            "hero_in": self.hero_in,
            "hero_out": self.hero_out,
            "frames": self.frames,
            "prompt_preview": self.prompt_preview,
            "negative_prompt_preview": self.negative_prompt_preview,
            "infer_shell": self.infer_shell,
            "extract_hero_shell": self.extract_hero_shell,
        }


@dataclass
class MinuteRenderPlan:
    out_dir: str
    segments: list[SegmentRenderStep]
    concat_shell: str
    full_output: str
    temporal_upscale: dict[str, Any]
    env: dict[str, Any]
    validation_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "full_output": self.full_output,
            "validation_issues": self.validation_issues,
            "env": self.env,
            "segments": [s.to_dict() for s in self.segments],
            "concat_shell": self.concat_shell,
            "temporal_upscale": self.temporal_upscale,
        }


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def _infer_command(
    env: InferEnv,
    *,
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    output_mp4: Path,
    condition_image: Path | None,
) -> str:
    neg = negative_prompt or "worst quality, inconsistent motion, blurry, jittery, distorted"
    if env.hq:
        parts = [
            "python",
            str(env.repo_root / "pipeline/two_stage_hq_kino.py"),
            "--checkpoint-path",
            str(env.native_ltx),
            "--gemma-root",
            str(env.gemma),
            "--prompt",
            prompt,
            "--negative-prompt",
            neg,
            "--output-path",
            str(output_mp4),
            "--distilled-lora",
            str(env.distilled_lora),
            "1.0",
            "--spatial-upsampler-path",
            str(env.spatial_upsampler),
            "--lora",
            str(env.lora_ckpt),
            "0.85",
            "--height",
            str(max(544, env.height * 2 if env.height <= 576 else env.height)),
            "--width",
            str(max(960, env.width * 2 if env.width <= 1024 else env.width)),
            "--num-frames",
            str(min(num_frames, env.hq_frames_cap)),
            "--frame-rate",
            str(env.fps),
            "--num-inference-steps",
            str(env.hq_steps),
            "--offload",
            os.environ.get("GOPEX_DELIVERY_HQ_OFFLOAD", "disk"),
            "--experimental-flat-dim-bridge",
        ]
        if condition_image is not None:
            parts.extend(["--images", str(condition_image)])
        if not env.with_audio:
            parts.append("--skip-audio")
        return _shell_join(parts)

    parts = [
        "python",
        str(env.trainer_dir / "scripts/inference.py"),
        "--checkpoint",
        str(env.native_ltx),
        "--text-encoder-path",
        str(env.gemma),
        "--lora-path",
        str(env.lora_ckpt),
        "--prompt",
        prompt,
        "--negative-prompt",
        neg,
        "--height",
        str(env.height),
        "--width",
        str(env.width),
        "--num-frames",
        str(num_frames),
        "--frame-rate",
        str(env.fps),
        "--num-inference-steps",
        str(env.inference_steps),
        "--guidance-scale",
        str(env.guidance_scale),
        "--output",
        str(output_mp4),
    ]
    if env.text_stack is not None and env.text_stack.is_file():
        parts.extend(["--text-stack-path", str(env.text_stack)])
    else:
        parts.extend(["--flat-dim-bridge-rank", str(env.bridge_rank)])
    if condition_image is not None:
        parts.extend(["--condition-image", str(condition_image)])
    if not env.with_audio:
        parts.append("--skip-audio")
    return _shell_join(parts)


def _extract_hero_command(video: Path, hero_png: Path) -> str:
    return _shell_join(
        [
            "ffmpeg",
            "-y",
            "-sseof",
            "-0.04",
            "-i",
            str(video),
            "-vframes",
            "1",
            str(hero_png),
        ]
    )


def build_minute_render_plan(
    compose: MinuteComposePlan,
    *,
    out_dir: str | Path,
    env: InferEnv | None = None,
    run_name: str = "minute",
) -> MinuteRenderPlan:
    """Build per-segment inference + concat plan from a compose plan."""
    infer_env = env or InferEnv.from_env()
    issues = infer_env.validate()
    root = Path(out_dir).expanduser().resolve()
    seg_dir = root / run_name / "segments"
    hero_dir = root / run_name / "heroes"
    list_file = root / run_name / "concat_list.txt"
    full_out = root / run_name / f"{run_name}_full.mp4"

    steps: list[SegmentRenderStep] = []
    prev_hero: Path | None = None

    for seg in compose.segments:
        out_mp4 = seg_dir / f"seg_{seg.index:02d}.mp4"
        hero_out: Path | None = None
        if seg.index < len(compose.segments) - 1:
            hero_out = hero_dir / f"hero_{seg.index:02d}.png"
        hero_in = prev_hero if seg.carry_hero and prev_hero is not None else None
        infer = _infer_command(
            infer_env,
            prompt=seg.prompt,
            negative_prompt=seg.negative_prompt,
            num_frames=seg.frames,
            output_mp4=out_mp4,
            condition_image=hero_in,
        )
        extract = _extract_hero_command(out_mp4, hero_out) if hero_out is not None else None
        steps.append(
            SegmentRenderStep(
                index=seg.index,
                output_mp4=str(out_mp4),
                hero_in=str(hero_in) if hero_in else None,
                hero_out=str(hero_out) if hero_out else None,
                infer_shell=infer,
                extract_hero_shell=extract,
                frames=seg.frames,
                prompt_preview=seg.prompt[:200],
                negative_prompt_preview=seg.negative_prompt[:200],
            )
        )
        prev_hero = hero_out

    concat_lines = [f"file '{s.output_mp4}'" for s in steps]
    list_body = "\n".join(concat_lines)
    concat_shell = (
        f"mkdir -p {shlex.quote(str(seg_dir))} {shlex.quote(str(hero_dir))} && "
        f"printf '%s\\n' {shlex.quote(list_body)} > {shlex.quote(str(list_file))} && "
        f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(list_file))} -c copy {shlex.quote(str(full_out))}"
    )

    return MinuteRenderPlan(
        out_dir=str(root),
        segments=steps,
        concat_shell=concat_shell,
        full_output=str(full_out),
        temporal_upscale=temporal_upscale_post_concat_command(input_mp4=str(full_out), output_mp4=str(full_out.with_name(full_out.stem + "_temporal_x2.mp4"))),
        env=infer_env.to_dict(),
        validation_issues=issues,
    )


def render_plan_from_prompt(
    *,
    prompt: str,
    target_duration_s: float = 60.0,
    fps: float = 24.0,
    out_dir: str | Path,
    env: InferEnv | None = None,
) -> MinuteRenderPlan:
    compose = compose_minute_plan(global_prompt=prompt, target_duration_s=target_duration_s, fps=fps)
    return build_minute_render_plan(compose, out_dir=out_dir, env=env, run_name="t2av")


def render_plan_from_case(
    case: BenchmarkCase,
    *,
    out_dir: str | Path,
    fps: float = 24.0,
    env: InferEnv | None = None,
) -> MinuteRenderPlan:
    compose = segment_from_events(case, fps=fps)
    return build_minute_render_plan(compose, out_dir=out_dir, env=env, run_name=case.case_id)


def render_plan_longav_example(*, out_dir: str | Path, env: InferEnv | None = None) -> MinuteRenderPlan:
    return render_plan_from_case(example_t2av_performance_ads_l4(), out_dir=out_dir, env=env)
