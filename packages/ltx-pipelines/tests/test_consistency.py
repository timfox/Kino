import argparse

import pytest

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_pipelines.consistency import consistency_arg_parser, validate_consistency_args
from ltx_pipelines.utils.helpers import (
    ConsistencyKeyframeInput,
    build_consistency_guider_params,
    build_consistency_plan,
    build_reference_aware_prompt,
)


def test_build_consistency_plan_adds_hero_and_keyframe_reinforcement() -> None:
    plan = build_consistency_plan(
        num_frames=97,
        hero_image_path="/refs/hero.png",
        keyframes=[ConsistencyKeyframeInput(path="/refs/alt.png", frame_idx=32)],
        reference_video_path=None,
        reference_mask_path=None,
    )

    assert plan.backend == "ti2vid"
    assert [image.frame_idx for image in plan.images] == [0, 32, 64, 96]
    assert plan.images[1].path == "/refs/alt.png"
    assert plan.images[2].path == "/refs/hero.png"


def test_build_consistency_plan_routes_reference_video_to_ic_lora() -> None:
    plan = build_consistency_plan(
        num_frames=121,
        hero_image_path="/refs/hero.png",
        keyframes=[],
        reference_video_path="/refs/clip.mp4",
        reference_mask_path="/refs/mask.mp4",
        preset_name="masked_subject",
        consistency_strength=0.5,
    )

    assert plan.backend == "ic_lora"
    assert plan.video_conditioning == (("/refs/clip.mp4", 0.4),)
    assert plan.conditioning_attention_mask_path == "/refs/mask.mp4"
    assert plan.conditioning_attention_strength == 0.5
    assert "masked subject region" in plan.prompt_reference_hint


def test_build_consistency_plan_requires_reference_video_for_mask() -> None:
    with pytest.raises(ValueError, match="reference_mask_path requires reference_video_path"):
        build_consistency_plan(
            num_frames=121,
            hero_image_path="/refs/hero.png",
            keyframes=[],
            reference_video_path=None,
            reference_mask_path="/refs/mask.mp4",
        )


def test_build_consistency_guider_params_applies_preset_to_video_only() -> None:
    video_params = MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    )
    audio_params = MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    )

    updated_video, updated_audio = build_consistency_guider_params(
        video_params,
        audio_params,
        preset_name="strong_identity",
        consistency_strength=1.0,
    )

    assert updated_video.cfg_scale == pytest.approx(3.6)
    assert updated_video.rescale_scale == pytest.approx(0.78)
    assert updated_video.modality_scale == pytest.approx(3.4)
    assert updated_audio == audio_params


def test_consistency_parser_accepts_high_level_reference_inputs() -> None:
    parser = consistency_arg_parser()
    args = parser.parse_args(
        [
            "--gemma-root",
            "/models/gemma",
            "--prompt",
            "A runner sprints through a stadium tunnel.",
            "--output-path",
            "/tmp/out.mp4",
            "--spatial-upsampler-path",
            "/models/upsampler.safetensors",
            "--hero-image",
            "/refs/hero.png",
            "--keyframe",
            "/refs/keyframe.png",
            "24",
            "--reference-video",
            "/refs/clip.mp4",
            "--reference-mask",
            "/refs/mask.mp4",
            "--distilled-checkpoint-path",
            "/models/distilled.safetensors",
            "--ic-lora",
            "/models/ic_lora.safetensors",
            "0.8",
        ]
    )

    assert args.hero_image == "/refs/hero.png"
    assert args.reference_video == "/refs/clip.mp4"
    assert args.reference_mask == "/refs/mask.mp4"
    assert len(args.consistency_keyframes) == 1
    assert args.consistency_keyframes[0].frame_idx == 24
    assert args.consistency_keyframes[0].strength is None
    assert args.ic_lora[0].path == "/models/ic_lora.safetensors"


def test_build_reference_aware_prompt_includes_hint() -> None:
    prompt = build_reference_aware_prompt(
        "A runner sprints through a stadium tunnel.",
        "Preserve the same athlete identity and uniform details from the references.",
    )

    assert "Preserve the same athlete identity" in prompt
    assert "Scene request: A runner sprints through a stadium tunnel." in prompt


def test_validate_consistency_args_requires_reference_input() -> None:
    args = argparse.Namespace(
        hero_image=None,
        consistency_keyframes=[],
        reference_video=None,
        reference_mask=None,
        images=[],
        checkpoint_path="/models/base.safetensors",
        distilled_checkpoint_path=None,
        distilled_lora=["/models/distilled.safetensors"],
        ic_lora=[],
    )

    with pytest.raises(ValueError, match="requires at least one reference input"):
        validate_consistency_args(args)


def test_validate_consistency_args_requires_backend_specific_artifacts() -> None:
    still_args = argparse.Namespace(
        hero_image="/refs/hero.png",
        consistency_keyframes=[],
        reference_video=None,
        reference_mask=None,
        images=[],
        checkpoint_path=None,
        distilled_checkpoint_path=None,
        distilled_lora=[],
        ic_lora=[],
    )
    reference_args = argparse.Namespace(
        hero_image="/refs/hero.png",
        consistency_keyframes=[],
        reference_video="/refs/clip.mp4",
        reference_mask=None,
        images=[],
        checkpoint_path=None,
        distilled_checkpoint_path=None,
        distilled_lora=[],
        ic_lora=[],
    )

    with pytest.raises(ValueError, match="--checkpoint-path is required"):
        validate_consistency_args(still_args)
    with pytest.raises(ValueError, match="--distilled-checkpoint-path is required"):
        validate_consistency_args(reference_args)
