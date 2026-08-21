from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, ValidationInfo, field_validator, model_validator

from ltx_trainer.project_sampling import ProjectSamplingMode
from ltx_trainer.quantization import QuantizationOptions
from ltx_trainer.training_strategies.base_strategy import TrainingStrategyConfigBase
from ltx_trainer.training_strategies.flexible import FlexibleStrategyConfig
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig
from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig
from ltx_trainer.training_strategies.audio_ref_only_ic import AudioRefOnlyICConfig


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model and training mode"""

    model_path: str | Path = Field(
        ...,
        description="Model path - local path to safetensors checkpoint file",
    )

    text_encoder_path: str | Path | None = Field(
        default=None,
        description="Path to text encoder (required for LTX-2/Gemma models, optional for LTXV/T5 models)",
    )

    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Training mode - either LoRA fine-tuning or full model fine-tuning",
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    gemma_encode_stack_dims: dict[str, int] | None = Field(
        default=None,
        description="Optional {'hidden_size', 'num_stack'} merged into Gemma HF config for the embeddings processor "
        "when metadata disagrees with actual encode() stacking (advanced; must match real hidden states).",
    )

    require_matched_gemma_text_flat_dim: bool = Field(
        default=False,
        description="If true, fail when Gemma stacked width (hidden_size×num_stack) differs from the LTX "
        "checkpoint's video_aggregate_embed.in_features instead of using the experimental flat_dim bridge. "
        "Set true for native Gemma 4 geometry (no bridge); keep false only when you intentionally use the bridge.",
    )

    flat_dim_bridge_rank: int | None = Field(
        default=None,
        ge=2,
        description="When Gemma flat width and the LTX checkpoint disagree and the experimental bridge is enabled, "
        "use a bottleneck Linear→Linear with this inner rank instead of one dense projection (much smaller; "
        "suitable if you later train the bridge). Omit for legacy dense random bridge.",
    )

    finetune_text_connectors: bool = Field(
        default=False,
        description="Include video (and audio) embeddings_connector weights in the optimizer. Gradients flow from "
        "the diffusion loss through precomputed caption features into the connectors. Saved as a sidecar "
        "text_embeds_weights_step_*.safetensors next to LoRA/full checkpoints. Validation prompts cached at startup "
        "still use the initial connector weights unless you disable that cache.",
    )

    finetune_text_stack: bool = Field(
        default=False,
        description="Train feature_extractor (flat_dim bridge + video/audio aggregate linears). Keeps feature_extractor "
        "on GPU; with text_stack_live_captions runs Gemma encode() each step. Saved as text_stack_weights_step_*.safetensors. "
        "Use text_stack_freeze_dit for Phase 1a (frozen DiT).",
    )

    text_stack_freeze_dit: bool = Field(
        default=True,
        description="When finetune_text_stack is true, do not train the diffusion transformer (Phase 1a). Set false to "
        "jointly train LoRA/full DiT and the text stack.",
    )

    text_stack_live_captions: bool = Field(
        default=True,
        description="When finetune_text_stack is true, load captions from data.dataset_manifest_path and run the full "
        "Gemma → feature_extractor → connector path each step (required to train bridge/aggregates).",
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str | Path) -> str | Path:
        """Validate that model_path is either a valid URL or an existing local path."""
        is_url = str(v).startswith(("http://", "https://"))

        if is_url:
            raise ValueError(f"Model path cannot be a URL: {v}")

        if not Path(v).exists():
            raise ValueError(f"Model path does not exist: {v}")

        return v

    @field_validator("gemma_encode_stack_dims")
    @classmethod
    def validate_gemma_encode_stack_dims(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        if v is None:
            return None
        allowed = {"hidden_size", "num_stack"}
        extra = set(v) - allowed
        if extra:
            raise ValueError(f"gemma_encode_stack_dims: unknown keys {sorted(extra)} (allowed: {sorted(allowed)})")
        if "hidden_size" not in v or "num_stack" not in v:
            raise ValueError("gemma_encode_stack_dims must include 'hidden_size' and 'num_stack'")
        return v


class LoraConfig(ConfigBaseModel):
    """Configuration for LoRA fine-tuning"""

    rank: int = Field(
        default=64,
        description="Rank of LoRA adaptation",
        ge=2,
    )

    alpha: int = Field(
        default=64,
        description="Alpha scaling factor for LoRA",
        ge=1,
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        le=1.0,
    )

    target_modules: list[str] = Field(
        default=["to_k", "to_q", "to_v", "to_out.0"],
        description="List of modules to target with LoRA",
    )


def _get_strategy_discriminator(v: dict | TrainingStrategyConfigBase) -> str:
    """Discriminator function for strategy config union."""
    if isinstance(v, dict):
        return v.get("name", "text_to_video")
    return v.name


# Union type for all strategy configs with discriminator
TrainingStrategyConfig = Annotated[
    Annotated[TextToVideoConfig, Tag("text_to_video")]
    | Annotated[VideoToVideoConfig, Tag("video_to_video")]
    | Annotated[FlexibleStrategyConfig, Tag("flexible")]
    | Annotated[AudioRefOnlyICConfig, Tag("audio_ref_only_ic")],
    Discriminator(_get_strategy_discriminator),
]


class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters"""

    learning_rate: float = Field(
        default=5e-4,
        description="Learning rate for optimization",
    )

    steps: int = Field(
        default=3000,
        description="Number of training steps",
    )

    batch_size: int = Field(
        default=2,
        description="Batch size for training",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
    )

    optimizer_type: Literal["adamw", "adamw8bit"] = Field(
        default="adamw",
        description="Type of optimizer to use for training",
    )

    scheduler_type: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "step",
    ] = Field(
        default="linear",
        description="Type of scheduler to use for training",
    )

    scheduler_params: dict = Field(
        default_factory=dict,
        description="Parameters for the scheduler",
    )

    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of slower training",
    )


class AccelerationConfig(ConfigBaseModel):
    """Configuration for hardware acceleration and compute optimization"""

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(
        default="bf16",
        description="Mixed precision training mode",
    )

    quantization: QuantizationOptions | None = Field(
        default=None,
        description="Quantization precision to use",
    )

    load_text_encoder_in_8bit: bool = Field(
        default=False,
        description="Whether to load the text encoder in 8-bit precision to save memory",
    )


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing"""

    preprocessed_data_root: str = Field(
        description="Path to folder containing preprocessed training data",
    )

    num_dataloader_workers: int = Field(
        default=2,
        description="Number of background processes for data loading (0 means synchronous loading)",
        ge=0,
    )

    dataset_manifest_path: str | Path | None = Field(
        default=None,
        description="dataset.json (or CSV/JSONL) with caption + media_path. Required when model.finetune_text_stack and "
        "model.text_stack_live_captions are enabled; used to align captions with precomputed latent shards.",
    )

    project_sampling_mode: ProjectSamplingMode = Field(
        default="uniform",
        description="How to weight clips across merged catalog projects: uniform (dataset size), "
        "balanced (equal gradient per project), sqrt (soft cap on dominant pools).",
    )


# =============================================================================
# Validation Condition Types
# =============================================================================


class FirstFrameConditionConfig(ConfigBaseModel):
    """First-frame conditioning (intrinsic, latent_idx=0). Always targets video.
    If image_or_video points to a video file, the first frame is automatically extracted.
    """

    type: Literal["first_frame"] = "first_frame"
    image_or_video: str | Path


class PrefixConditionConfig(ConfigBaseModel):
    """Prefix conditioning for temporal extension (intrinsic). Exactly one of video/audio must be set."""

    type: Literal["prefix"] = "prefix"
    video: str | None = None
    audio: str | None = None
    num_frames: int | None = Field(
        default=None,
        ge=1,
        description="Number of pixel frames for video prefix. Must satisfy num_frames %% 8 == 1.",
    )
    duration: float | None = Field(default=None, gt=0, description="Duration in seconds for audio prefix")

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "PrefixConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for prefix condition")
        return self

    @model_validator(mode="after")
    def validate_num_frames_constraint(self) -> "PrefixConditionConfig":
        if self.video is not None and self.num_frames is not None and self.num_frames % 8 != 1:
            raise ValueError(
                f"num_frames ({self.num_frames}) must satisfy num_frames % 8 == 1 "
                f"for video prefix (e.g., 1, 9, 17, 25, ...)"
            )
        return self


class SuffixConditionConfig(ConfigBaseModel):
    """Suffix conditioning for temporal extension (intrinsic). Exactly one of video/audio must be set."""

    type: Literal["suffix"] = "suffix"
    video: str | None = None
    audio: str | None = None
    num_frames: int | None = Field(
        default=None,
        ge=1,
        description="Number of pixel frames for video suffix. Must satisfy num_frames %% 8 == 0.",
    )
    duration: float | None = Field(default=None, gt=0, description="Duration in seconds for audio suffix")

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "SuffixConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for suffix condition")
        return self

    @model_validator(mode="after")
    def validate_num_frames_constraint(self) -> "SuffixConditionConfig":
        if self.video is not None and self.num_frames is not None and self.num_frames % 8 != 0:
            raise ValueError(
                f"num_frames ({self.num_frames}) must satisfy num_frames % 8 == 0 "
                f"for video suffix (e.g., 8, 16, 24, 32, ...)"
            )
        return self


class SpatialCropConditionConfig(ConfigBaseModel):
    """Spatial crop conditioning for outpainting (intrinsic, video only)."""

    type: Literal["spatial_crop"] = "spatial_crop"
    video: str
    spatial_region: tuple[int, int, int, int] = Field(
        ..., description="Spatial crop region as (y1, x1, y2, x2) in pixel coordinates"
    )


class MaskConditionConfig(ConfigBaseModel):
    """Mask-based conditioning for inpainting (intrinsic). Exactly one of video/audio must be set."""

    type: Literal["mask"] = "mask"
    video: str | None = None
    audio: str | None = None
    mask: str

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "MaskConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for mask condition")
        return self


class ReferenceConditionConfig(ConfigBaseModel):
    """Reference conditioning (IC-LoRA style concatenation). Exactly one of video/audio must be set."""

    type: Literal["reference"] = "reference"
    video: str | None = None
    audio: str | None = None
    downscale_factor: int = Field(default=1, ge=1)
    temporal_scale_factor: int = Field(default=1, ge=1)
    include_in_output: bool = False

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "ReferenceConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for reference condition")
        return self


class VideoToAudioConditionConfig(ConfigBaseModel):
    """Video-to-audio — video is provided as frozen cross-modal conditioning.
    The video is kept clean (sigma=0) and influences audio generation via cross-modal attention.
    """

    type: Literal["video_to_audio"] = "video_to_audio"
    video: str


class AudioToVideoConditionConfig(ConfigBaseModel):
    """Audio-to-video — audio is provided as frozen cross-modal conditioning.
    The audio is kept clean (sigma=0) and influences video generation via cross-modal attention.
    """

    type: Literal["audio_to_video"] = "audio_to_video"
    audio: str


ValidationCondition = Annotated[
    Union[
        FirstFrameConditionConfig,
        PrefixConditionConfig,
        SuffixConditionConfig,
        SpatialCropConditionConfig,
        MaskConditionConfig,
        ReferenceConditionConfig,
        VideoToAudioConditionConfig,
        AudioToVideoConditionConfig,
    ],
    Field(discriminator="type"),
]


def _condition_targets_video(cond: ValidationCondition) -> bool:
    """Check if a validation condition targets the video modality."""
    if cond.type in ("first_frame", "spatial_crop", "video_to_audio"):
        return True
    if cond.type in ("prefix", "suffix", "mask", "reference"):
        return getattr(cond, "video", None) is not None
    return False


def _condition_targets_audio(cond: ValidationCondition) -> bool:
    """Check if a validation condition targets the audio modality."""
    if cond.type == "audio_to_video":
        return True
    if cond.type in ("prefix", "suffix", "mask", "reference"):
        return getattr(cond, "audio", None) is not None
    return False


class ValidationSample(ConfigBaseModel):
    """Configuration for a single validation sample — fully self-describing."""

    prompt: str
    conditions: list[ValidationCondition] = Field(default_factory=list)

    video_dims: tuple[int, int, int] | None = Field(
        default=None,
        description="Per-sample override for (width, height, frames). None = inherit from ValidationConfig.",
    )
    seed: int | None = Field(
        default=None,
        description="Per-sample override for random seed. None = inherit from ValidationConfig.",
    )

    @field_validator("video_dims")
    @classmethod
    def validate_video_dims(cls, v: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        if v is None:
            return v
        width, height, frames = v
        if width % 32 != 0:
            raise ValueError(f"Width ({width}) must be divisible by 32")
        if height % 32 != 0:
            raise ValueError(f"Height ({height}) must be divisible by 32")
        if frames % 8 != 1:
            raise ValueError(f"Frames ({frames}) must satisfy frames % 8 == 1 for LTX-2 (e.g., 1, 9, 17, 25, ...)")
        return v

    @model_validator(mode="after")
    def validate_frozen_modality_conflicts(self) -> "ValidationSample":
        frozen_types = {c.type for c in self.conditions if c.type in ("video_to_audio", "audio_to_video")}

        if "video_to_audio" in frozen_types and "audio_to_video" in frozen_types:
            raise ValueError(
                "Cannot have both video_to_audio and audio_to_video conditions — nothing would be generated"
            )

        if "video_to_audio" in frozen_types:
            for c in self.conditions:
                if c.type != "video_to_audio" and _condition_targets_video(c):
                    raise ValueError(
                        f"Cannot use video-targeting '{c.type}' condition when video is frozen (video_to_audio)"
                    )

        if "audio_to_video" in frozen_types:
            for c in self.conditions:
                if c.type != "audio_to_video" and _condition_targets_audio(c):
                    raise ValueError(
                        f"Cannot use audio-targeting '{c.type}' condition when audio is frozen (audio_to_video)"
                    )

        return self


class ValidationConfig(ConfigBaseModel):
    """Configuration for validation during training"""

    samples: list[ValidationSample] = Field(
        default_factory=list,
        description="Per-sample validation config (preferred). Replaces legacy prompts/images/reference_videos.",
    )

    prompts: list[str] = Field(
        default_factory=list,
        description="[DEPRECATED: use samples] List of prompts to use for validation",
    )

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt to use for validation examples",
    )

    images: list[str] | None = Field(
        default=None,
        description="[DEPRECATED: use samples with first_frame conditions] List of image paths to use for validation. "
        "One image path must be provided for each validation prompt",
    )

    reference_videos: list[str] | None = Field(
        default=None,
        description="[DEPRECATED: use samples with reference conditions] List of reference video paths to use for validation. "
        "One video path must be provided for each validation prompt",
    )

    reference_downscale_factor: int = Field(
        default=1,
        description="[DEPRECATED: use downscale_factor on ReferenceCondition] Downscale factor for reference videos in IC-LoRA validation. "
        "When > 1, reference videos are processed at 1/n resolution (e.g., 2 means half resolution). "
        "Must match the factor used during dataset preprocessing.",
        ge=1,
    )

    video_dims: tuple[int, int, int] = Field(
        default=(960, 544, 97),
        description="Dimensions of validation videos (width, height, frames). "
        "Width and height must be divisible by 32. Frames must satisfy frames % 8 == 1 for LTX-2.",
    )

    @field_validator("video_dims")
    @classmethod
    def validate_video_dims(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        """Validate video dimensions for LTX-2 compatibility."""
        width, height, frames = v

        if width % 32 != 0:
            raise ValueError(f"Width ({width}) must be divisible by 32")
        if height % 32 != 0:
            raise ValueError(f"Height ({height}) must be divisible by 32")
        if frames % 8 != 1:
            raise ValueError(f"Frames ({frames}) must satisfy frames % 8 == 1 for LTX-2 (e.g., 1, 9, 17, 25, ...)")

        return v

    frame_rate: float = Field(
        default=25.0,
        description="Frame rate for validation videos",
        gt=0,
    )

    seed: int = Field(
        default=42,
        description="Random seed used when sampling validation videos",
    )

    inference_steps: int = Field(
        default=50,
        description="Number of inference steps for validation",
        gt=0,
    )

    interval: int | None = Field(
        default=100,
        description="Number of steps between validation runs. If None, validation is disabled.",
        gt=0,
    )

    guidance_scale: float = Field(
        default=4.0,
        description="CFG guidance scale to use during validation",
        ge=1.0,
    )

    stg_scale: float = Field(
        default=1.0,
        description="STG (Spatio-Temporal Guidance) scale. 0.0 disables STG. "
        "Recommended value is 1.0. STG is combined with CFG for improved video quality.",
        ge=0.0,
    )

    stg_blocks: list[int] | None = Field(
        default=[29],
        description="Which transformer blocks to perturb for STG. "
        "None means all blocks are perturbed. Recommended for LTX-2: [29].",
    )

    stg_mode: Literal["stg_av", "stg_v"] = Field(
        default="stg_av",
        description="STG mode: 'stg_av' skips both audio and video self-attention, "
        "'stg_v' skips only video self-attention.",
    )

    generate_audio: bool = Field(
        default=True,
        description="Whether to generate audio in validation samples. "
        "Independent of training strategy setting - you can generate audio "
        "in validation even when not training the audio branch.",
    )

    generate_video: bool = Field(
        default=True,
        description="Whether to generate video in validation samples. Set False for audio-only validation.",
    )

    skip_initial_validation: bool = Field(
        default=False,
        description="Skip validation video sampling at step 0 (beginning of training)",
    )

    include_reference_in_output: bool = Field(
        default=False,
        description="[DEPRECATED: use include_in_output on ReferenceCondition] For video-to-video training: concatenate the original reference video side-by-side "
        "with the generated output. The reference comes from the input video, not from the model's output.",
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of images (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of images ({len(v)}) must match number of prompts ({num_prompts})")

        for image_path in v:
            if not Path(image_path).exists():
                raise ValueError(f"Image path '{image_path}' does not exist")

        return v

    @field_validator("reference_videos")
    @classmethod
    def validate_reference_videos(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of reference videos (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of reference videos ({len(v)}) must match number of prompts ({num_prompts})")

        for video_path in v:
            if not Path(video_path).exists():
                raise ValueError(f"Reference video path '{video_path}' does not exist")

        return v


    @model_validator(mode="after")
    def convert_legacy_format(self) -> "ValidationConfig":
        """Convert deprecated prompts/images/reference_videos to samples format."""
        if self.prompts and not self.samples:
            samples = []
            for i, prompt in enumerate(self.prompts):
                conditions: list[ValidationCondition] = []
                if self.images and i < len(self.images):
                    conditions.append(FirstFrameConditionConfig(image_or_video=self.images[i]))
                if self.reference_videos and i < len(self.reference_videos):
                    conditions.append(
                        ReferenceConditionConfig(
                            video=self.reference_videos[i],
                            downscale_factor=self.reference_downscale_factor,
                            include_in_output=self.include_reference_in_output,
                        )
                    )
                samples.append(ValidationSample(prompt=prompt, conditions=conditions))
            self.samples = samples
        return self

    @model_validator(mode="after")
    def validate_scaled_reference_dimensions(self) -> "ValidationConfig":
        """Validate that scaled reference dimensions are valid when reference_downscale_factor > 1."""
        if self.reference_downscale_factor > 1:
            width, height, _frames = self.video_dims

            # Validate that downscale factor evenly divides the target dimensions
            if width % self.reference_downscale_factor != 0:
                raise ValueError(
                    f"Width {width} is not evenly divisible by reference_downscale_factor "
                    f"{self.reference_downscale_factor}. Choose a downscale factor that divides {width} evenly."
                )
            if height % self.reference_downscale_factor != 0:
                raise ValueError(
                    f"Height {height} is not evenly divisible by reference_downscale_factor "
                    f"{self.reference_downscale_factor}. Choose a downscale factor that divides {height} evenly."
                )

            scaled_width = width // self.reference_downscale_factor
            scaled_height = height // self.reference_downscale_factor

            # Validate scaled dimensions are divisible by 32
            if scaled_width % 32 != 0:
                raise ValueError(
                    f"Scaled reference width {scaled_width} (from {width} / {self.reference_downscale_factor}) "
                    f"is not divisible by 32. Choose a different downscale factor or adjust video_dims."
                )
            if scaled_height % 32 != 0:
                raise ValueError(
                    f"Scaled reference height {scaled_height} (from {height} / {self.reference_downscale_factor}) "
                    f"is not divisible by 32. Choose a different downscale factor or adjust video_dims."
                )

        return self

    @model_validator(mode="after")
    def validate_output_modality_requirements(self) -> "ValidationConfig":
        has_validation = bool(self.prompts) or bool(self.samples)
        if has_validation and not self.generate_video and not self.generate_audio:
            raise ValueError(
                "At least one of generate_video or generate_audio must be True when validation is configured."
            )
        return self



class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training"""

    interval: int | None = Field(
        default=None,
        description="Number of steps between checkpoint saves. If None, intermediate checkpoints are disabled.",
        gt=0,
    )

    keep_last_n: int = Field(
        default=1,
        description="Number of most recent checkpoints to keep. Set to -1 to keep all checkpoints.",
        ge=-1,
    )

    precision: Literal["bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Precision to use when saving checkpoint weights. Options: 'bfloat16' or 'float32'.",
    )

    no_resume: bool = Field(
        default=False,
        description="When True, ignore any saved training state and start from step 0. "
        "Model weights from load_checkpoint are still loaded, but optimizer/scheduler "
        "state and step counter are reset.",
    )

    save_training_state: Literal["full", "minimal", "off"] = Field(
        default="minimal",
        description="Save training state alongside checkpoints for resume. "
        "'full': optimizer + scheduler + RNG + step (~800MB for LoRA, much larger for full fine-tuning). "
        "'minimal': scheduler + RNG + step only (~few KB, sufficient for LoRA). "
        "'off': nothing saved, resume not possible.",
    )


class HubConfig(ConfigBaseModel):
    """Configuration for Hugging Face Hub integration"""

    push_to_hub: bool = Field(default=False, description="Whether to push the model weights to the Hugging Face Hub")
    hub_model_id: str | None = Field(
        default=None, description="Hugging Face Hub repository ID (e.g., 'username/repo-name')"
    )

    @model_validator(mode="after")
    def validate_hub_config(self) -> "HubConfig":
        """Validate that hub_model_id is not None when push_to_hub is True."""
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")
        return self


class WandbConfig(ConfigBaseModel):
    """Configuration for Weights & Biases logging"""

    enabled: bool = Field(
        default=False,
        description="Whether to enable W&B logging",
    )

    project: str = Field(
        default="ltxv-trainer",
        description="W&B project name",
    )

    entity: str | None = Field(
        default=None,
        description="W&B username or team",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags to add to the W&B run",
    )

    log_validation_videos: bool = Field(
        default=True,
        description="Whether to log validation videos to W&B",
    )


class AvFoldTrainingConfig(ConfigBaseModel):
    """Use preprocess fold sidecars (``GOPEX_AV_FOLD_HOOKS``) during Phase 2 training."""

    enabled: bool = Field(
        default=False,
        description="Apply VSR quality weights and optional audio sidecar regularizers from .pt metadata",
    )
    require_sidecars: bool = Field(
        default=True,
        description=(
            "Hard-fail the first training step when av_fold is enabled but the batch has no fold "
            "sidecars (override with GOPEX_AV_FOLD_ALLOW_EMPTY=1)"
        ),
    )
    normalize_weight_product: bool = Field(
        default=True,
        description=(
            "If the product of per-sample fold multipliers collapses below min_weight_product, "
            "rescale so effective batch mass stays usable"
        ),
    )
    min_weight_product: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Floor on mean(product of fold weights) before optional rescale",
    )
    use_vsr_quality_weights: bool = Field(
        default=True,
        description="Downweight diffusion loss when vsr_vqa mos_proxy is low (needs preprocess fold)",
    )
    use_erp_boiqa_weights: bool = Field(
        default=True,
        description="Downweight loss when vuga/vuboiqa mos_proxy is low (360 ERP BOIQA fold sidecars)",
    )
    min_erp_boiqa_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum per-sample loss multiplier for lowest ERP BOIQA proxy",
    )
    use_s3po_wss_weights: bool = Field(
        default=True,
        description="Downweight loss when s3po wss_quality_proxy is low (360 ERP VSR fold sidecars)",
    )
    min_s3po_wss_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest S3PO WSS quality proxy",
    )
    use_lucky_hdr_weights: bool = Field(
        default=True,
        description="Upweight clips with high lucky_hdr merge_readiness_proxy (HDR bracket sidecars)",
    )
    min_lucky_hdr_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest LuckyHDR merge readiness",
    )
    use_era_defocus_weights: bool = Field(
        default=False,
        description="Upweight clips with high era_defocus deblur_readiness/sharpness (defocus fold; enable for blur datasets)",
    )
    min_era_defocus_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest ErA defocus sharpness/readiness",
    )
    use_flood_physics_weights: bool = Field(
        default=False,
        description="Upweight clips with high flood_physics inundation_readiness (physics-guided flood fold; off by default — not cinematic)",
    )
    min_flood_physics_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest flood physics inundation readiness",
    )
    use_dilated_sym_diff_weights: bool = Field(
        default=False,
        description="Upweight clips with high dilated_sym_diff change_detection_readiness (binary compare fold; off by default)",
    )
    min_dilated_sym_diff_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest change-detection readiness",
    )
    use_dynamic_gp_weights: bool = Field(
        default=False,
        description="Upweight clips with high dynamic_gp filter_readiness (evolving-function / IDE fold; off by default)",
    )
    min_dynamic_gp_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest dynamic GP spatiotemporal readiness",
    )
    use_branch_energy_weights: bool = Field(
        default=False,
        description="Upweight clips with high branch_energy localization_readiness (power-theory fold; off by default)",
    )
    min_branch_energy_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest branch-level energy localization readiness",
    )
    use_lora_hd_attn_weights: bool = Field(
        default=False,
        description="Upweight clips with high lora_hd_attn rank-one readiness (HD LoRA theory fold; off by default)",
    )
    min_lora_hd_attn_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest HD LoRA attention readiness",
    )
    use_latenthdr_weights: bool = Field(
        default=True,
        description="Upweight clips with high latenthdr exposure_readiness_proxy (HDR EV bracket sidecars)",
    )
    min_latenthdr_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest LatentHDR exposure readiness",
    )
    use_sdr2hdr_weights: bool = Field(
        default=True,
        description="Upweight clips with high sdr2hdr merge_readiness_proxy (SDR→HDR MEVM sidecars)",
    )
    min_sdr2hdr_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest SDR2HDR merge readiness",
    )
    use_x2hdr_weights: bool = Field(
        default=True,
        description="Upweight clips with high x2hdr pu21_alignment_readiness",
    )
    min_x2hdr_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest X2HDR PU21 alignment readiness",
    )
    use_lf_diff_weights: bool = Field(
        default=True,
        description="Upweight clips with high lf_diff bracket_readiness",
    )
    min_lf_diff_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest LF-Diff bracket readiness",
    )
    use_diffhdr_weights: bool = Field(
        default=True,
        description="Upweight clips with high diffhdr log_gamma_readiness",
    )
    min_diffhdr_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest DiffHDR log-gamma readiness",
    )
    use_vdp_hdr_weights: bool = Field(
        default=True,
        description="Upweight clips with high vdp_hdr fusion_readiness_proxy",
    )
    min_vdp_hdr_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest VDP-HDR fusion readiness",
    )
    use_pantheon360_weights: bool = Field(
        default=True,
        description="Upweight 360° clips with high pantheon360 cache_fusion_readiness",
    )
    min_pantheon360_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest Pantheon360 cache fusion readiness",
    )
    use_mirage_weights: bool = Field(
        default=True,
        description="Upweight clips with high mirage readout_coverage (latent spatial memory proxy)",
    )
    min_mirage_readout_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest Mirage readout coverage",
    )
    use_semantic_stitch_weights: bool = Field(
        default=True,
        description="Upweight stitched ERP skies with high semantic_stitch seam_coherence",
    )
    min_semantic_stitch_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest SemanticStitch seam coherence",
    )
    use_sphere360_weights: bool = Field(
        default=True,
        description="Upweight clips with high sphere360 timelapse_readiness (360° Hub sidecars)",
    )
    min_sphere360_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest Sphere360 timelapse readiness",
    )
    use_panoworld_weights: bool = Field(
        default=True,
        description="Upweight ERP clips with high panoworld panospace_readiness",
    )
    min_panoworld_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest PanoWorld panospace readiness",
    )
    use_gimbal360_weights: bool = Field(
        default=True,
        description="Upweight ERP clips with high gimbal360 canonical_erp_readiness",
    )
    min_gimbal360_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest Gimbal360 canonical ERP readiness",
    )
    use_panoenv_weights: bool = Field(
        default=True,
        description="Upweight ERP clips with high panoenv spatial_vqa_readiness",
    )
    min_panoenv_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest PanoEnv spatial VQA readiness",
    )
    use_weatherproof_weights: bool = Field(
        default=True,
        description="Upweight clips with labeled weather degradation (weatherproof robustness_readiness)",
    )
    min_weatherproof_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest WeatherProof robustness readiness",
    )
    use_dense360_weights: bool = Field(
        default=True,
        description="Upweight ERP clips with high dense360 omni_vlm_readiness",
    )
    min_dense360_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest Dense360 omni-VLM readiness",
    )
    use_cross360_weights: bool = Field(
        default=True,
        description="Upweight ERP clips with high cross360 depth_fusion_readiness",
    )
    min_cross360_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest Cross360 depth fusion readiness",
    )
    use_anything360_weights: bool = Field(
        default=True,
        description="Upweight ERP clips with high anything360 conditioning_readiness",
    )
    min_anything360_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest 360Anything conditioning readiness",
    )
    use_spherefusion_weights: bool = Field(
        default=True,
        description="Upweight ERP clips with high spherefusion depth_readiness",
    )
    min_spherefusion_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest SphereFusion depth readiness",
    )
    use_raim_mef_weights: bool = Field(
        default=True,
        description="Upweight HDR brackets with high raim_mef mef_fusion_readiness",
    )
    min_raim_mef_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest RAIM MEF fusion readiness",
    )
    use_lumivid_weights: bool = Field(
        default=True,
        description="Upweight HDR clips with high lumivid logc3_alignment_readiness",
    )
    min_lumivid_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest LumiVid LogC3 alignment readiness",
    )
    use_stem2_sdr_hdr_weights: bool = Field(
        default=True,
        description="Upweight cinema HDR sidecars with high stem2 isotonic_readiness_proxy",
    )
    min_stem2_sdr_hdr_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest StEM2 isotonic readiness",
    )
    use_physthdr_gs_weights: bool = Field(
        default=True,
        description="Upweight multi-exposure clips with high physthdr_gs gs_training_readiness",
    )
    min_physthdr_gs_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest PhysHDR-GS training readiness",
    )
    use_modulo_spike_hdr_weights: bool = Field(
        default=True,
        description="Upweight spike-sensor clips with high modulo_spike_hdr unwrap_readiness_proxy",
    )
    min_modulo_spike_hdr_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest modulo-spike HDR unwrap readiness",
    )
    use_cubediff_weights: bool = Field(default=True, description="Upweight cubemap ERP clips with high cubediff panorama_readiness")
    min_cubediff_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum cubediff panorama readiness multiplier")
    use_spherediff_weights: bool = Field(default=True, description="Upweight ERP clips with high spherediff spherical_readiness")
    min_spherediff_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum spherediff spherical readiness multiplier")
    use_erpgs_weights: bool = Field(default=True, description="Upweight ERP clips with high erpgs splat_readiness")
    min_erpgs_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum ERP-GS splat readiness multiplier")
    use_sphereuformer_weights: bool = Field(default=True, description="Upweight 360° clips with high sphereuformer perception_readiness")
    min_sphereuformer_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum SphereUFormer perception readiness multiplier")
    use_sphere_depth_weights: bool = Field(default=True, description="Upweight ERP clips with high sphere_depth calibration_readiness")
    min_sphere_depth_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum Sphere-Depth calibration readiness multiplier")
    use_nvc_erp_qpa_weights: bool = Field(default=True, description="Upweight ERP clips with high nvc_erp_qpa qpa_readiness")
    min_nvc_erp_qpa_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum NVC ERP QPA readiness multiplier")
    use_mtpano_weights: bool = Field(default=True, description="Upweight ERP clips with high mtpano multitask_dense_readiness")
    min_mtpano_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum MTPano multitask readiness multiplier")
    use_panoworld_x_weights: bool = Field(default=True, description="Upweight ERP clips with high panoworld_x explorable_world_readiness")
    min_panoworld_x_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum PanoWorld-X explorable world readiness multiplier")
    use_pano_affordance_weights: bool = Field(default=True, description="Upweight ERP clips with high pano_affordance affordance_readiness")
    min_pano_affordance_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum Pano-Affordance readiness multiplier")
    use_panogsdet_weights: bool = Field(default=True, description="Upweight ERP clips with high panogsdet gs_detection_readiness")
    min_panogsdet_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum PanoGSDet GS detection readiness multiplier")
    use_panolm_weights: bool = Field(default=True, description="Upweight ERP clips with high panolm panovqa_readiness")
    min_panolm_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum PanoLM pano-VQA readiness multiplier")
    use_pano_flight_weights: bool = Field(default=True, description="Upweight aerial ERP clips with high pano_flight survey_task_readiness")
    min_pano_flight_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum Pano Flight survey task readiness multiplier")
    use_pano360_weights: bool = Field(default=True, description="Upweight ERP clips with high pano360 photogrammetry_readiness")
    min_pano360_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum pano360 photogrammetry readiness multiplier")
    use_holitok_weights: bool = Field(default=True, description="Upweight audio shards with high holitok holistic_readiness_proxy")
    min_holitok_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum HoliTok holistic readiness multiplier")
    use_flatsounds_weights: bool = Field(default=True, description="Upweight V2A clips with high flatsounds onset_strength_proxy")
    min_flatsounds_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum FlatSounds onset strength multiplier")
    min_quality_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum per-sample loss multiplier for lowest MOS proxy",
    )
    downweight_degraded_audio: bool = Field(
        default=True,
        description="Downweight loss when speech_quality_emb degraded_frame_ratio is high",
    )
    use_speech_mos_weights: bool = Field(
        default=True,
        description="Downweight audio-video loss when speech_quality_emb pseudo_mos_mean is low",
    )
    min_audio_quality_weight: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum audio-quality loss multiplier for lowest pseudo MOS",
    )
    degraded_audio_penalty: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Strength of degraded-audio downweighting",
    )
    downweight_unstable_audio: bool = Field(
        default=True,
        description="Downweight loss when robustspeechflow repeat_delta_mean is high",
    )
    unstable_audio_penalty: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Strength of alignment-instability downweighting",
    )
    lambda_fmelcodec_coding: float = Field(
        default=0.0,
        ge=0.0,
        description="Add mean fmelcodec coding_loss_proxy (audio shards only; usually 0)",
    )
    lambda_wavenext2_spectral: float = Field(
        default=0.0,
        ge=0.0,
        description="Add WaveNeXt 2 spectral envelope stability proxy from audio sidecars",
    )
    downweight_low_physics: bool = Field(
        default=True,
        description="Downweight loss when phyworld physics_proxy is low (needs preprocess fold)",
    )
    use_avbench_alignment_weights: bool = Field(
        default=True,
        description="Downweight loss when avbench av_align_proxy is low (short T2AV alignment proxy)",
    )
    min_avbench_weight: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest AVBench alignment proxy",
    )
    use_mmae_instruction_weights: bool = Field(
        default=False,
        description="Upweight loss when MMAE edit_complexity_proxy is high (instruction-edit sidecar)",
    )
    min_mmae_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest MMAE edit complexity proxy",
    )
    use_mtavg2_expressiveness_weights: bool = Field(
        default=False,
        description="Downweight loss when mtavg2 expressiveness_risk_proxy is high (cinematic failure risk)",
    )
    min_mtavg2_weight: float = Field(
        default=0.42,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for highest MTAVG2 expressiveness risk",
    )
    mtavg2_risk_penalty: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Strength of high expressiveness_risk downweighting",
    )
    use_growloop_weights: bool = Field(
        default=False,
        description="Upweight audio loss when growloop final_score_bucket is high (human-likeness proxy)",
    )
    min_growloop_weight: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum audio loss multiplier for lowest GrowLoop score bucket",
    )
    low_physics_penalty: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Strength of low physics_proxy downweighting",
    )
    use_fuse_flow_weights: bool = Field(
        default=True,
        description="Upweight clips with high fuse_flow fusion_readiness_proxy (metric multi-view)",
    )
    min_fuse_flow_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum loss multiplier for lowest FUSE fusion readiness",
    )
    downweight_low_geometry_stability: bool = Field(
        default=True,
        description="Downweight loss when fuse_flow geometry_stability_proxy is low",
    )
    low_geometry_penalty: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Strength of low geometry_stability_proxy downweighting",
    )
    lambda_fuse_flow_geometry: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional batch-level geometry stability target loss (fuse_flow sidecars)",
    )
    fuse_flow_geometry_target: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Target mean geometry_stability_proxy when lambda_fuse_flow_geometry > 0",
    )
    log_lamo_motion_drift: bool = Field(
        default=True,
        description="Log lamo motion_drift_norm mean when sidecars present (no loss change)",
    )
    lambda_lamo_drift: float = Field(
        default=0.0,
        ge=0.0,
        description="Add mean lamo motion_drift_norm as auxiliary loss (motion stability)",
    )
    lambda_phyworld_dpo: float = Field(
        default=0.0,
        ge=0.0,
        description="PhyWorld DPO surrogate from physics_proxy pairwise ranking (needs B>=2 or accum)",
    )
    phyworld_dpo_beta: float = Field(
        default=100.0,
        ge=1.0,
        description="β for diffusion_dpo_loss in physics ranking surrogate",
    )
    lambda_phyworld_proxy_target: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional MSE pull of batch mean physics_proxy toward 4.0",
    )
    lambda_group_action_temporal: float = Field(
        default=0.0,
        ge=0.0,
        description="Group-action identity loss between first/last latent frames",
    )
    log_bernini_segments: bool = Field(
        default=True,
        description="Log bernini num_segments when sidecars present",
    )
    log_adamag_guidance_norm: bool = Field(
        default=True,
        description="Log adamag guidance_norm_proxy from fold sidecars at train time",
    )
    downweight_high_core_kd_drift: bool = Field(
        default=False,
        description="Downweight loss when core_kd state_drift_proxy is high (missing-modality MER proxy)",
    )
    core_kd_drift_penalty: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Strength of core_kd state_drift downweighting (normalized drift in ~0..1)",
    )
    downweight_entroad_anomaly: bool = Field(
        default=False,
        description="Downweight loss when entroad localized_entropy_spike_proxy is high",
    )
    entroad_anomaly_penalty: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Strength of entroad anomaly-spike downweighting",
    )
    use_eigenet_reverb_weights: bool = Field(
        default=False,
        description="Upweight audio loss when eigenet decay_ratio_proxy suggests stable reverberation",
    )
    min_eigenet_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum audio multiplier for lowest eigenet decay_ratio_proxy",
    )
    use_planaudio_composition_weights: bool = Field(
        default=False,
        description="Upweight audio when planaudio semantic_coverage_factor_proxy is high",
    )
    min_planaudio_weight: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum audio multiplier for lowest planaudio semantic coverage",
    )
    use_ag_repa_causal_weights: bool = Field(
        default=False,
        description="Upweight audio loss when ag_repa fog_a_proxy is high (FoG-A causal bottleneck)",
    )
    min_ag_repa_weight: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum audio multiplier for lowest FoG-A proxy",
    )
    ag_repa_fog_scale: float = Field(
        default=0.2,
        gt=0.0,
        description="Normalize fog_a_proxy to ~0..1 (typical early-layer FoG-A scores)",
    )
    use_forte_t2a_weights: bool = Field(
        default=True,
        description="Upweight video loss when forte t2a_align_proxy is high (FORTE T2A sidecar)",
    )
    min_forte_t2a_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum video multiplier for lowest forte t2a_align_proxy",
    )
    use_forte_audio_weights: bool = Field(
        default=True,
        description="Upweight audio loss when forte_audio t2a_align_proxy is high",
    )
    min_forte_audio_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum audio multiplier for lowest forte_audio t2a_align_proxy",
    )
    use_omnicustom_video_weights: bool = Field(
        default=False,
        description="Upweight video loss when omnicustom identity_proxy is high (speech-span ready)",
    )
    min_omnicustom_identity_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum video multiplier for lowest omnicustom identity_proxy",
    )
    use_omnicustom_timbre_weights: bool = Field(
        default=False,
        description="Upweight audio loss when omnicustom timbre_proxy is high",
    )
    min_omnicustom_timbre_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum audio multiplier for lowest omnicustom timbre_proxy",
    )
    use_id_lora_video_weights: bool = Field(
        default=False,
        description="Upweight video when id_lora identity_proxy / structured [VISUAL]+[SPEECH] prompt is ready",
    )
    min_id_lora_identity_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum video multiplier for lowest id_lora identity_proxy",
    )
    use_id_lora_audio_weights: bool = Field(
        default=False,
        description="Upweight audio when id_lora speaker_identity_proxy is high",
    )
    min_id_lora_speaker_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum audio multiplier for lowest id_lora speaker_identity_proxy",
    )
    use_autocut_video_weights: bool = Field(
        default=False,
        description="Upweight video loss when autocut narrative_proxy / edit_ready_proxy is high",
    )
    min_autocut_narrative_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum video multiplier for lowest autocut narrative_proxy",
    )
    use_autocut_bgm_weights: bool = Field(
        default=False,
        description="Upweight audio loss when autocut bgm_proxy is high (BGM selection readiness)",
    )
    min_autocut_bgm_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum audio multiplier for lowest autocut bgm_proxy",
    )
    use_svhighlights_saliency_weights: bool = Field(
        default=True,
        description="Upweight video loss when svhighlights saliency_norm is high (TF-SELECTOR sidecar)",
    )
    min_svhighlights_saliency_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum video multiplier for lowest svhighlights saliency_norm",
    )
    log_research_fold_metrics: bool = Field(
        default=True,
        description="Log core_kd / vlm_count / entroad / flatsounds / mtavg2 / ag_repa / omnicustom / autocut sidecar means",
    )
    log_metrics: bool = Field(
        default=True,
        description="Log av_fold/* scalars to W&B when sidecars are used",
    )


class LatentHdrConfig(ConfigBaseModel):
    """LatentHDR-style exposure head (joint L_diff + λ L_ev). Requires HDR preprocess shards."""

    enabled: bool = Field(
        default=False,
        description="Add L_ev from FiLMResidualExposureHead when latents include hdr_ldr_ev_stack",
    )
    lambda_ev: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight on exposure latent MSE (LatentHDR paper: joint with diffusion loss)",
    )
    exposure_head_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to train_latenthdr_exposure.py output .pt (state_dict + latent_channels)",
    )
    train_exposure_head: bool = Field(
        default=False,
        description="Include exposure head in optimizer (usually false after phase-1 head train)",
    )
    latent_channels: int = Field(
        default=128,
        ge=8,
        description="VAE latent channel count when building a fresh head (ignored when loading checkpoint)",
    )


class FlowMatchingConfig(ConfigBaseModel):
    """Configuration for flow matching training"""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(
        default="shifted_logit_normal",
        description="Mode to use for timestep sampling",
    )

    timestep_sampling_params: dict = Field(
        default_factory=dict,
        description="Parameters for timestep sampling",
    )


class LtxTrainerConfig(ConfigBaseModel):
    """Unified configuration for LTXV training"""

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = Field(default=None)
    training_strategy: TrainingStrategyConfig = Field(
        default_factory=TextToVideoConfig,
        description="Training strategy configuration. Determines the training mode and its parameters.",
    )
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    latenthdr: LatentHdrConfig = Field(default_factory=LatentHdrConfig)
    av_fold: AvFoldTrainingConfig = Field(default_factory=AvFoldTrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    # General configuration
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    output_dir: str = Field(
        default="outputs",
        description="Directory to save model outputs",
    )

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        """Expand user home directory in output path."""
        return str(Path(v).expanduser().resolve())

    def _validate_data_dirs_exist(self) -> None:
        data_root = Path(self.data.preprocessed_data_root)
        for dir_name in self.training_strategy.get_data_sources():
            dir_path = data_root / dir_name
            if not dir_path.is_dir():
                raise ValueError(
                    f"Required data directory '{dir_name}' does not exist under preprocessed_data_root: {dir_path}"
                )

    @model_validator(mode="after")
    def validate_strategy_compatibility(self) -> "LtxTrainerConfig":
        """Validate that training strategy and other configurations are compatible."""
        self._validate_data_dirs_exist()

        # Check that reference videos are provided when using video_to_video strategy
        if self.training_strategy.name == "video_to_video" and self.validation.interval:
            has_reference = bool(self.validation.reference_videos) or any(
                cond.type == "reference" for sample in self.validation.samples for cond in sample.conditions
            )
            if not has_reference:
                raise ValueError(
                    "reference_videos or samples with reference conditions must be provided "
                    "in validation config when using video_to_video strategy"
                )

        # Check that LoRA config is provided when training mode is lora
        if self.model.training_mode == "lora" and self.lora is None:
            raise ValueError("LoRA configuration must be provided when training_mode is 'lora'")

        # Check that LoRA config is provided when using video_to_video or audio_ref_only_ic strategy
        if self.training_strategy.name in {"video_to_video", "audio_ref_only_ic"} and self.model.training_mode != "lora":
            raise ValueError(
                f"Training mode must be 'lora' when using {self.training_strategy.name} strategy"
            )

        if self.latenthdr.enabled:
            if not self.latenthdr.exposure_head_checkpoint and not self.latenthdr.train_exposure_head:
                raise ValueError(
                    "latenthdr.enabled requires exposure_head_checkpoint or train_exposure_head=true"
                )

        if self.model.finetune_text_stack:
            if not self.model.text_encoder_path:
                raise ValueError("model.text_encoder_path is required when model.finetune_text_stack is true")
            if self.model.text_stack_live_captions and not self.data.dataset_manifest_path:
                raise ValueError(
                    "data.dataset_manifest_path is required when finetune_text_stack and text_stack_live_captions "
                    "are enabled"
                )
            if self.model.text_stack_freeze_dit and self.model.training_mode == "full":
                raise ValueError("text_stack_freeze_dit cannot be used with training_mode 'full'")

        return self
