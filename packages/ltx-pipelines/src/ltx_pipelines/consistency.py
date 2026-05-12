import argparse
import logging
from collections.abc import Iterator

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import (
    DEFAULT_IMAGE_CRF,
    ImageConditioningInput,
    LoraAction,
    default_1_stage_arg_parser,
    resolve_path,
)
from ltx_pipelines.utils.constants import LTX_2_3_PARAMS, PipelineParams, detect_params
from ltx_pipelines.utils.helpers import (
    CONSISTENCY_PRESETS,
    DEFAULT_CONSISTENCY_PRESET,
    ConsistencyKeyframeInput,
    build_consistency_guider_params,
    build_consistency_plan,
    build_reference_aware_prompt,
    load_mask_video,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode


class ConsistencyKeyframeAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) not in (2, 3, 4):
            msg = (
                f"{option_string} requires 2 to 4 arguments "
                "(PATH FRAME_IDX [STRENGTH] [CRF])"
            )
            raise argparse.ArgumentError(self, msg)

        current = getattr(namespace, self.dest) or []
        current.append(
            ConsistencyKeyframeInput(
                path=resolve_path(values[0]),
                frame_idx=int(values[1]),
                strength=float(values[2]) if len(values) >= 3 else None,
                crf=int(values[3]) if len(values) == 4 else DEFAULT_IMAGE_CRF,
            )
        )
        setattr(namespace, self.dest, current)


def detect_consistency_params() -> PipelineParams:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--checkpoint-path", type=resolve_path, required=False)
    pre.add_argument("--distilled-checkpoint-path", type=resolve_path, required=False)
    known, _ = pre.parse_known_args()
    if known.checkpoint_path:
        return detect_params(known.checkpoint_path)
    if known.distilled_checkpoint_path:
        return detect_params(known.distilled_checkpoint_path)
    return LTX_2_3_PARAMS


def consistency_arg_parser(params: PipelineParams = LTX_2_3_PARAMS) -> argparse.ArgumentParser:
    parser = default_1_stage_arg_parser(params=params)
    for action in parser._actions:
        if "--checkpoint-path" in action.option_strings:
            action.required = False
            action.help = "Path to the full LTX-2 model checkpoint. Required when no reference video is supplied."
        if "--image" in action.option_strings:
            action.help = (
                "Advanced image override: PATH FRAME_IDX STRENGTH [CRF]. "
                "Prefer --hero-image and --keyframe for the consistency workflow."
            )

    parser.add_argument(
        "--distilled-checkpoint-path",
        type=resolve_path,
        default=None,
        help="Path to the distilled LTX-2 model checkpoint. Required when --reference-video is used.",
    )
    parser.add_argument(
        "--distilled-lora",
        dest="distilled_lora",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="Distilled LoRA for the TI2Vid refinement stage. Required when no reference video is supplied.",
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=resolve_path,
        required=True,
        help="Path to the spatial upsampler model used by both consistency backends.",
    )
    parser.add_argument(
        "--hero-image",
        type=resolve_path,
        default=None,
        help="Primary subject reference image used to lock identity from frame 0 onward.",
    )
    parser.add_argument(
        "--keyframe",
        dest="consistency_keyframes",
        action=ConsistencyKeyframeAction,
        nargs="+",
        metavar="ARG",
        default=[],
        help=(
            "Identity anchor keyframe: PATH FRAME_IDX [STRENGTH] [CRF]. "
            "When strength is omitted the selected consistency preset chooses it."
        ),
    )
    parser.add_argument(
        "--reference-video",
        type=resolve_path,
        default=None,
        help="Optional reference clip. When provided, the wrapper dispatches to ICLoraPipeline.",
    )
    parser.add_argument(
        "--reference-mask",
        type=resolve_path,
        default=None,
        help="Optional grayscale subject mask video paired with --reference-video.",
    )
    parser.add_argument(
        "--ic-lora",
        dest="ic_lora",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="IC-LoRA weights used when the consistency wrapper dispatches to ICLoraPipeline.",
    )
    parser.add_argument(
        "--consistency-preset",
        choices=sorted(CONSISTENCY_PRESETS),
        default=DEFAULT_CONSISTENCY_PRESET,
        help="Identity-retention preset controlling default strengths and reinforcement behavior.",
    )
    parser.add_argument(
        "--consistency-strength",
        type=float,
        default=1.0,
        help="Global multiplier applied to the chosen consistency preset.",
    )
    return parser


def validate_consistency_args(args: argparse.Namespace) -> None:
    has_any_reference = bool(args.hero_image or args.consistency_keyframes or args.reference_video or args.images)
    if not has_any_reference:
        raise ValueError(
            "ConsistencyPipeline requires at least one reference input: "
            "--hero-image, --keyframe, --reference-video, or --image."
        )

    if args.reference_mask and not args.reference_video:
        raise ValueError("--reference-mask requires --reference-video.")

    if args.reference_video:
        if args.distilled_checkpoint_path is None:
            raise ValueError("--distilled-checkpoint-path is required when using --reference-video.")
        if not args.ic_lora:
            raise ValueError("--ic-lora is required when using --reference-video.")
        return

    if args.checkpoint_path is None:
        raise ValueError("--checkpoint-path is required when using still-reference consistency mode.")
    if not args.distilled_lora:
        raise ValueError("--distilled-lora is required when using still-reference consistency mode.")


class ConsistencyPipeline:
    """High-level identity-consistency wrapper over TI2Vid and IC-LoRA pipelines."""

    def __init__(
        self,
        checkpoint_path: str | None,
        distilled_checkpoint_path: str | None,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        ic_loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
        offload_mode: OffloadMode = OffloadMode.NONE,
    ):
        self.checkpoint_path = checkpoint_path
        self.distilled_checkpoint_path = distilled_checkpoint_path
        self.distilled_lora = distilled_lora
        self.spatial_upsampler_path = spatial_upsampler_path
        self.gemma_root = gemma_root
        self.loras = loras
        self.ic_loras = ic_loras
        self.device = device
        self.quantization = quantization
        self.registry = registry
        self.torch_compile = torch_compile
        self.offload_mode = offload_mode

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        hero_image_path: str | None,
        keyframes: list[ConsistencyKeyframeInput],
        reference_video_path: str | None,
        reference_mask_path: str | None,
        consistency_preset: str = DEFAULT_CONSISTENCY_PRESET,
        consistency_strength: float = 1.0,
        images: list[ImageConditioningInput] | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        max_batch_size: int = 1,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        plan = build_consistency_plan(
            num_frames=num_frames,
            hero_image_path=hero_image_path,
            keyframes=keyframes,
            reference_video_path=reference_video_path,
            reference_mask_path=reference_mask_path,
            preset_name=consistency_preset,
            consistency_strength=consistency_strength,
            advanced_images=images,
        )
        logging.info(
            "ConsistencyPipeline dispatch=%s images=%d reference_videos=%d",
            plan.backend,
            len(plan.images),
            len(plan.video_conditioning),
        )
        prompt = build_reference_aware_prompt(prompt, plan.prompt_reference_hint)

        if plan.backend == "ti2vid":
            if self.checkpoint_path is None:
                raise ValueError("--checkpoint-path is required when no reference video is supplied")
            if not self.distilled_lora:
                raise ValueError("--distilled-lora is required when no reference video is supplied")

            video_guider_params, audio_guider_params = build_consistency_guider_params(
                video_guider_params,
                audio_guider_params,
                preset_name=consistency_preset,
                consistency_strength=consistency_strength,
            )
            pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=self.checkpoint_path,
                distilled_lora=self.distilled_lora,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=self.gemma_root,
                loras=self.loras,
                device=self.device,
                quantization=self.quantization,
                registry=self.registry,
                torch_compile=self.torch_compile,
                offload_mode=self.offload_mode,
            )
            return pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=list(plan.images),
                tiling_config=tiling_config,
                enhance_prompt=enhance_prompt,
                max_batch_size=max_batch_size,
            )

        if self.distilled_checkpoint_path is None:
            raise ValueError("--distilled-checkpoint-path is required when using --reference-video")
        if not self.ic_loras:
            raise ValueError("--ic-lora is required when using --reference-video")

        conditioning_attention_mask = None
        if plan.conditioning_attention_mask_path is not None:
            conditioning_attention_mask = load_mask_video(
                mask_path=plan.conditioning_attention_mask_path,
                height=height // 2,
                width=width // 2,
                num_frames=num_frames,
            )

        pipeline = ICLoraPipeline(
            distilled_checkpoint_path=self.distilled_checkpoint_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            gemma_root=self.gemma_root,
            loras=self.ic_loras,
            device=self.device,
            quantization=self.quantization,
            registry=self.registry,
            torch_compile=self.torch_compile,
            offload_mode=self.offload_mode,
        )
        return pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=list(plan.images),
            video_conditioning=list(plan.video_conditioning),
            enhance_prompt=enhance_prompt,
            tiling_config=tiling_config,
            conditioning_attention_strength=plan.conditioning_attention_strength,
            conditioning_attention_mask=conditioning_attention_mask,
        )


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    params = detect_consistency_params()
    parser = consistency_arg_parser(params=params)
    args = parser.parse_args()
    validate_consistency_args(args)

    pipeline = ConsistencyPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        distilled_lora=list(args.distilled_lora),
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=list(args.lora),
        ic_loras=list(args.ic_lora),
        quantization=args.quantization,
        torch_compile=args.compile,
        offload_mode=args.offload_mode,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        hero_image_path=args.hero_image,
        keyframes=args.consistency_keyframes,
        reference_video_path=args.reference_video,
        reference_mask_path=args.reference_mask,
        consistency_preset=args.consistency_preset,
        consistency_strength=args.consistency_strength,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        max_batch_size=args.max_batch_size,
    )
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
