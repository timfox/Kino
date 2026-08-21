#!/usr/bin/env python3

"""
Compute latent representations for video generation training.
This module provides functionality for processing video and image files, including:
- Loading videos/images from various file formats (CSV, JSON, JSONL)
- Resizing, cropping, and transforming media
- MediaDataset for video-only preprocessing workflows
- BucketSampler for grouping videos by resolution
Can be used as a standalone script:
    python scripts/process_videos.py dataset.csv --resolution-buckets 768x768x25 \
        --output-dir /path/to/output --model-source /path/to/ltx2.safetensors
"""

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
import typer
from accelerate import PartialState
from pillow_heif import register_heif_opener
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, resize, to_tensor
from torchvision.transforms.functional import resize as tv_resize
from transformers.utils.logging import disable_progress_bar

from ltx_core.model.audio_vae import AudioProcessor
from ltx_core.types import Audio
from ltx_trainer import logger
from ltx_trainer.hdr_ingest import (
    ev_list_arange,
    hdr_meta_dict_to_padded_u8,
    latenthdr_meta_block,
    lf_diff_meta_block,
    linear_scene_to_lf_diff_tonemap_display,
    linear_scene_to_logc3_display,
    linear_scene_to_pu21_display,
    lumivid_meta_block,
    padded_u8_to_hdr_meta_dict,
    parse_ev_bracket_spec,
    read_video_hdr_float32,
    reinhard_tonemap,
    synthetic_gamma_ldr_stack_from_linear_hdr,
    x2hdr_meta_block,
)
from ltx_trainer.model_loader import load_audio_vae_encoder, load_video_vae_encoder
from ltx_trainer.utils import open_image_as_srgb
from ltx_trainer.video_utils import get_video_frame_count, read_video

disable_progress_bar()

# Register HEIF/HEIC support
register_heif_opener()

# Constants for validation
VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8

# Audio constants
AUDIO_LATENT_CHANNELS = 8
AUDIO_FREQUENCY_BINS = 16

DEFAULT_TILE_SIZE = 512  # Spatial tile size in pixels (must be ≥64 and divisible by 32)
DEFAULT_TILE_OVERLAP = 128  # Spatial tile overlap in pixels (must be divisible by 32)

_ALLOWED_HDR_TRANSFER = frozenset({"auto", "pq", "hlg", "srgb", "linear"})
_ALLOWED_HDR_VAE_ENCODING = frozenset({"reinhard", "pu21", "logc3", "lf_log1p"})


def _default_with_audio_from_env() -> bool:
    return os.environ.get("GOPEX_PREP_WITH_AUDIO", "1").strip().lower() not in ("0", "false", "no", "off")


def _parse_latent_save_dtype(name: str) -> torch.dtype:
    key = name.lower().strip()
    mapping = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if key not in mapping:
        raise typer.BadParameter(
            f"Unknown latent save dtype {name!r}; expected one of: {', '.join(sorted(mapping))}"
        )
    return mapping[key]


app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process videos/images and save latent representations for video generation training.",
)


def _clamp_01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_(0, 1)


class MediaDataset(Dataset):
    """
    Dataset for processing video and image files.
    This dataset is designed for media preprocessing workflows where you need to:
    - Load and preprocess videos/images
    - Apply resizing and cropping transformations
    - Handle different resolution buckets
    - Filter out invalid media files
    - Optionally extract audio from video files
    """

    def __init__(
        self,
        dataset_file: str | Path,
        main_media_column: str,
        video_column: str,
        resolution_buckets: list[tuple[int, int, int]],
        reshape_mode: str = "center",
        with_audio: bool = False,
        temporal_subsample_factor: int = 1,
        existing_video_shards: set[str] | None = None,
        existing_audio_shards: set[str] | None = None,
        filter_existing_shards: bool = False,
        hdr_ingest: bool = False,
        hdr_transfer: str = "auto",
        hdr_synth_bracket_ev: str | None = None,
        hdr_vae_encoding: str = "reinhard",
    ) -> None:
        """
        Initialize the media dataset.
        Args:
            dataset_file: Path to CSV/JSON/JSONL metadata file
            video_column: Column name for video paths in the metadata file
            resolution_buckets: List of (frames, height, width) tuples
            reshape_mode: How to crop videos ("center", "random")
            with_audio: Whether to extract audio from video files
            temporal_subsample_factor: Factor for VAE-aligned temporal subsampling.
                When > 1, keeps frame 0 then takes every Nth frame from frame 1 onwards.
            hdr_ingest: If True, decode video to scene-linear float32 and store ``hdr_latent``.
            hdr_transfer: Color transfer for HDR linearization when ``hdr_ingest`` is True.
            hdr_synth_bracket_ev: Optional EV bracket spec for synthetic LDR stack (requires hdr_ingest).
            hdr_vae_encoding: Tone-map / VAE input encoding when hdr_ingest is True.
        """
        super().__init__()

        self.dataset_file = Path(dataset_file)
        self.main_media_column = main_media_column
        self.resolution_buckets = resolution_buckets
        self.reshape_mode = reshape_mode
        self.with_audio = with_audio
        self.temporal_subsample_factor = temporal_subsample_factor
        self.existing_video_shards = existing_video_shards or set()
        self.existing_audio_shards = existing_audio_shards or set()
        self.filter_existing_shards = filter_existing_shards
        self.hdr_ingest = hdr_ingest
        self.hdr_transfer = hdr_transfer
        self.hdr_synth_bracket_ev = hdr_synth_bracket_ev
        hve = hdr_vae_encoding.lower().strip()
        if hve not in _ALLOWED_HDR_VAE_ENCODING:
            raise ValueError(
                f"Unknown hdr_vae_encoding {hdr_vae_encoding!r}; expected one of: "
                f"{', '.join(sorted(_ALLOWED_HDR_VAE_ENCODING))}"
            )
        self.hdr_vae_encoding = hve

        # Manifest-relative paths (e.g. ltx_manifest/clips/foo.mp4) drive latent shard layout.
        self.main_media_paths, self.main_media_relpaths = self._load_media_column(main_media_column)
        self.video_paths, self.video_relpaths = self._load_media_column(video_column)

        self._filter_valid_videos()

        self.max_target_frames = max(self.resolution_buckets, key=lambda x: x[0])[0]

        # Set up video transforms
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(_clamp_01),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single video/image with metadata, and optionally audio."""
        if isinstance(index, list):
            # Special case for BucketSampler - return cached data
            return index

        video_path: Path = self.video_paths[index]
        relative_path = str(self.video_relpaths[index])
        media_relative_path = str(self.main_media_relpaths[index])

        if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            media_tensor = self._preprocess_image(video_path)
            fps = 1.0
            audio_data = None  # Images don't have audio
            hdr_latent = None
            hdr_meta_u8 = None
            hdr_ldr_ev_stack = None
        elif self.audio_only_rows[index]:
            media_tensor, fps, audio_data = self._load_audio_only_item(video_path)
            hdr_latent = None
            hdr_meta_u8 = None
            hdr_ldr_ev_stack = None
        else:
            media_tensor, fps, hdr_latent, hdr_meta_u8, hdr_ldr_ev_stack = self._preprocess_video(video_path)

            # Extract audio if enabled
            if self.with_audio:
                # Calculate target duration from the processed video frames
                # This ensures audio is trimmed to match the exact video duration
                # media_tensor is [C, F, H, W] so shape[1] is num_frames
                target_duration = media_tensor.shape[1] / fps
                audio_data = self._extract_audio(video_path, target_duration)
            else:
                audio_data = None

        # media_tensor is [C, F, H, W] format for VAE compatibility
        _, num_frames, height, width = media_tensor.shape

        result = {
            "video": media_tensor,
            "relative_path": relative_path,
            "main_media_relative_path": media_relative_path,
            "video_metadata": {
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "fps": fps,
            },
        }

        if hdr_latent is not None and hdr_meta_u8 is not None:
            result["hdr_latent"] = hdr_latent
            result["hdr_meta_u8"] = hdr_meta_u8
            if hdr_ldr_ev_stack is not None:
                result["hdr_ldr_ev_stack"] = hdr_ldr_ev_stack

        # Add audio data if available
        if audio_data is not None:
            result["audio"] = audio_data

        return result

    @staticmethod
    def _extract_audio(video_path: Path, target_duration: float) -> dict[str, torch.Tensor | int] | None:
        """Extract audio track from a video file, trimmed/padded to match video duration."""
        audio = _load_audio_from_file(video_path, max_duration=target_duration)
        if audio is None:
            return None

        # Pad if shorter than target (_load_audio_from_file only trims, doesn't pad)
        target_samples = int(target_duration * audio.sampling_rate)
        if audio.waveform.shape[-1] < target_samples:
            padding = target_samples - audio.waveform.shape[-1]
            waveform = torch.nn.functional.pad(audio.waveform, (0, padding))
            logger.warning(f"Padded audio to {target_duration:.2f} seconds for {video_path}")
        else:
            waveform = audio.waveform

        return {"waveform": waveform, "sample_rate": audio.sampling_rate}

    def _load_media_column(self, column: str) -> tuple[list[Path], list[Path]]:
        """Load resolved filesystem paths and manifest-relative media paths."""
        paths, relpaths = _load_media_column_from_dataset(self.dataset_file, column)
        invalid = [p for p in paths if not p.is_file()]
        if invalid:
            raise ValueError(f"Found {len(invalid)} invalid paths in '{column}'. First few: {invalid[:5]}")
        return paths, relpaths

    def _load_audio_only_item(
        self, video_path: Path
    ) -> tuple[torch.Tensor, float, dict[str, torch.Tensor | int] | None]:
        """Resume audio encode when video latents already exist (skip video decode)."""
        bucket_frames, target_height, target_width = min(self.resolution_buckets, key=lambda x: x[0])
        from ltx_trainer.ffmpeg_io import probe_media

        probe = probe_media(video_path)
        fps = probe.video.avg_fps if probe.video else 24.0
        frame_count = get_video_frame_count(video_path)
        effective_frames = max(1, min(frame_count, bucket_frames))
        target_duration = effective_frames / max(fps, 1e-6)
        audio_data = self._extract_audio(video_path, target_duration) if self.with_audio else None
        media_tensor = torch.zeros(3, bucket_frames, target_height, target_width)
        return media_tensor, fps, audio_data

    def _filter_valid_videos(self) -> None:
        """Filter out videos with insufficient frames."""
        original_length = len(self.video_paths)
        valid_video_paths: list[Path] = []
        valid_video_relpaths: list[Path] = []
        valid_main_media_paths: list[Path] = []
        valid_main_media_relpaths: list[Path] = []
        audio_only_rows: list[bool] = []
        min_frames_required = min(self.resolution_buckets, key=lambda x: x[0])[0]

        for i, video_path in enumerate(self.video_paths):
            if self.filter_existing_shards and self.existing_video_shards:
                if _latent_shard_done(
                    self.main_media_relpaths[i],
                    existing_video=self.existing_video_shards,
                    existing_audio=self.existing_audio_shards,
                    require_audio=self.with_audio,
                ):
                    continue

                if _latent_shard_done(
                    self.main_media_relpaths[i],
                    existing_video=self.existing_video_shards,
                    existing_audio=self.existing_audio_shards,
                    require_audio=False,
                ):
                    valid_video_paths.append(video_path)
                    valid_video_relpaths.append(self.video_relpaths[i])
                    valid_main_media_paths.append(self.main_media_paths[i])
                    valid_main_media_relpaths.append(self.main_media_relpaths[i])
                    audio_only_rows.append(True)
                    continue

            if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                valid_video_paths.append(video_path)
                valid_video_relpaths.append(self.video_relpaths[i])
                valid_main_media_paths.append(self.main_media_paths[i])
                valid_main_media_relpaths.append(self.main_media_relpaths[i])
                audio_only_rows.append(False)
                continue

            try:
                frame_count = get_video_frame_count(video_path)

                if frame_count >= min_frames_required:
                    valid_video_paths.append(video_path)
                    valid_video_relpaths.append(self.video_relpaths[i])
                    valid_main_media_paths.append(self.main_media_paths[i])
                    valid_main_media_relpaths.append(self.main_media_relpaths[i])
                    audio_only_rows.append(False)
                else:
                    logger.warning(
                        f"Skipping video at {video_path} - has {frame_count} frames, "
                        f"which is less than the minimum required frames ({min_frames_required})"
                    )
            except Exception as e:
                logger.warning(f"Failed to read video at {video_path}: {e!s}")

        # Update path lists to maintain synchronization
        self.video_paths = valid_video_paths
        self.video_relpaths = valid_video_relpaths
        self.main_media_paths = valid_main_media_paths
        self.main_media_relpaths = valid_main_media_relpaths
        self.audio_only_rows = audio_only_rows

        dropped = original_length - len(self.video_paths)
        if dropped:
            audio_only_count = sum(audio_only_rows)
            logger.warning(
                f"Filtered dataset from {original_length} to {len(self.video_paths)} rows "
                f"({dropped} dropped: insufficient frames and/or existing shards complete; "
                f"{audio_only_count} audio-only resume)."
            )

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        """Preprocess a single image by resizing and applying transforms."""
        image = open_image_as_srgb(path)
        image = to_tensor(image)
        image = image.unsqueeze(0)  # Add frame dimension [1, C, H, W] for bucket selection

        # Find nearest resolution bucket and resize
        nearest_bucket = self._get_resolution_bucket_for_item(image)
        _, target_height, target_width = nearest_bucket
        image_resized = self._resize_and_crop(image, target_height, target_width)
        # _resize_and_crop returns [C, H, W] for single-frame input (squeeze removes dim 0)

        # Apply transforms
        image = self.transforms(image_resized)  # [C, H, W] -> [C, H, W]

        # Add frame dimension in VAE format: [C, H, W] -> [C, 1, H, W]
        image = image.unsqueeze(1)
        return image

    def _preprocess_video(
        self, path: Path
    ) -> tuple[torch.Tensor, float, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Preprocess a video by loading, resizing, and applying transforms.
        Returns:
            Tuple of (VAE input ``[C,F,H,W]``, fps, optional ``hdr_latent``, optional ``hdr_meta_u8``,
            optional ``hdr_ldr_ev_stack`` ``[N,C,F,H,W]`` γ-encoded synthetic LDRs when configured).
        """
        if self.hdr_ingest:
            video_linear, fps, hdr_meta = read_video_hdr_float32(
                path, max_frames=self.max_target_frames, hdr_transfer=self.hdr_transfer
            )
            nearest_bucket = self._get_resolution_bucket_for_item(video_linear)
            target_num_frames, target_height, target_width = nearest_bucket
            frames_linear = self._resize_and_crop(video_linear, target_height, target_width)
            frames_linear = frames_linear[:target_num_frames]
            if self.temporal_subsample_factor > 1:
                indices = _compute_temporal_subsample_indices(target_num_frames, self.temporal_subsample_factor)
                frames_linear = frames_linear[indices]
            hdr_latent = frames_linear.permute(1, 0, 2, 3).contiguous().to(torch.float32)

            if self.hdr_vae_encoding == "pu21":
                tonemapped = linear_scene_to_pu21_display(frames_linear)
            elif self.hdr_vae_encoding == "logc3":
                tonemapped = linear_scene_to_logc3_display(frames_linear)
            elif self.hdr_vae_encoding == "lf_log1p":
                tonemapped = linear_scene_to_lf_diff_tonemap_display(frames_linear)
            else:
                tonemapped = reinhard_tonemap(frames_linear)
            video = torch.stack([self.transforms(frame) for frame in tonemapped], dim=0)
            video = video.permute(1, 0, 2, 3).contiguous()

            ld_stack: torch.Tensor | None = None
            hdr_meta = {**hdr_meta, **latenthdr_meta_block(save_ldr_stack=bool(self.hdr_synth_bracket_ev))}
            if self.hdr_vae_encoding == "pu21":
                hdr_meta = {**hdr_meta, **x2hdr_meta_block(vae_encoding="pu21")}
            elif self.hdr_vae_encoding == "logc3":
                hdr_meta = {**hdr_meta, **lumivid_meta_block(vae_encoding="logc3")}
            elif self.hdr_vae_encoding == "lf_log1p":
                hdr_meta = {**hdr_meta, **lf_diff_meta_block()}
            hdr_meta["hdr_vae_encoding"] = self.hdr_vae_encoding
            if self.hdr_synth_bracket_ev:
                emin, emax, estep = parse_ev_bracket_spec(self.hdr_synth_bracket_ev)
                ld_stack, evs = synthetic_gamma_ldr_stack_from_linear_hdr(hdr_latent, emin, emax, estep)
                hdr_meta["latenthdr"]["synthetic_bracket_recipe"] = {
                    "ev_min": emin,
                    "ev_max": emax,
                    "ev_step": estep,
                    "gamma": 2.2,
                }
                hdr_meta["latenthdr"]["ev_list"] = evs
                hdr_meta["latenthdr"]["ev_spec_saved"] = self.hdr_synth_bracket_ev

            meta_u8 = hdr_meta_dict_to_padded_u8(hdr_meta)
            return video, fps, hdr_latent, meta_u8, ld_stack

        # Load video frames up to max_target_frames
        video, fps = read_video(path, max_frames=self.max_target_frames)

        nearest_bucket = self._get_resolution_bucket_for_item(video)
        target_num_frames, target_height, target_width = nearest_bucket
        frames_resized = self._resize_and_crop(video, target_height, target_width)

        # Trim video to target number of frames
        frames_resized = frames_resized[:target_num_frames]

        # VAE-aligned temporal subsampling: keep frame 0, then every Nth frame
        if self.temporal_subsample_factor > 1:
            indices = _compute_temporal_subsample_indices(target_num_frames, self.temporal_subsample_factor)
            frames_resized = frames_resized[indices]

        # Apply transforms to each frame and stack
        video = torch.stack([self.transforms(frame) for frame in frames_resized], dim=0)

        # Permute [F,C,H,W] -> [C,F,H,W] for VAE compatibility
        # After DataLoader batching, this becomes [B,C,F,H,W] which VAE expects
        video = video.permute(1, 0, 2, 3).contiguous()

        return video, fps, None, None, None

    def _get_resolution_bucket_for_item(self, media_tensor: torch.Tensor) -> tuple[int, int, int]:
        """Get the nearest resolution bucket for the given media tensor."""
        num_frames, _, height, width = media_tensor.shape

        def distance(bucket: tuple[int, int, int]) -> tuple:
            bucket_num_frames, bucket_height, bucket_width = bucket
            # Lexicographic key:
            # 1) minimize aspect-ratio diff (in log-scale, for invariance to shorter/longer ARs)
            # 2) prefer buckets with more frames (by using negative)
            # 3) prefer buckets with larger spatial area (by using negative)
            return (
                abs(math.log(width / height) - math.log(bucket_width / bucket_height)),
                -bucket_num_frames,
                -(bucket_height * bucket_width),
            )

        # Keep only buckets with <= available frames
        relevant_buckets = [b for b in self.resolution_buckets if b[0] <= num_frames]
        if not relevant_buckets:
            raise ValueError(f"No resolution buckets have <= {num_frames} frames. Available: {self.resolution_buckets}")

        # Find the bucket with the minimal distance (according to the function above) to the media item's shape.
        nearest_bucket = min(relevant_buckets, key=distance)

        return nearest_bucket

    def _resize_and_crop(self, media_tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Resize and crop tensor to target size."""
        # Get current dimensions
        current_height, current_width = media_tensor.shape[2], media_tensor.shape[3]

        # Calculate aspect ratios to determine which dimension to resize first
        current_aspect = current_width / current_height
        target_aspect = target_width / target_height

        # Resize while maintaining aspect ratio - scale to make the smaller dimension fit
        if current_aspect > target_aspect:
            # Current is wider than target, so scale by height
            new_width = int(current_width * target_height / current_height)
            media_tensor = resize(
                media_tensor,
                size=[target_height, new_width],  # type: ignore
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            # Current is taller than target, so scale by width
            new_height = int(current_height * target_width / current_width)
            media_tensor = resize(
                media_tensor,
                size=[new_height, target_width],
                interpolation=InterpolationMode.BICUBIC,
            )

        # Update dimensions after resize
        current_height, current_width = media_tensor.shape[2], media_tensor.shape[3]
        media_tensor = media_tensor.squeeze(0)

        # Calculate how much we need to crop from each dimension
        delta_h = current_height - target_height
        delta_w = current_width - target_width

        # Determine crop position based on reshape mode
        if self.reshape_mode == "random":
            # Random crop position
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif self.reshape_mode == "center":
            # Center crop
            top, left = delta_h // 2, delta_w // 2
        else:
            raise ValueError(f"Unsupported reshape mode: {self.reshape_mode}")

        # Perform the final crop to exact target dimensions
        media_tensor = crop(media_tensor, top=top, left=left, height=target_height, width=target_width)
        return media_tensor


def _compute_temporal_subsample_indices(num_frames: int, factor: int) -> list[int]:
    """Compute VAE-aligned temporal subsample indices.
    Keeps frame 0 (the VAE's standalone first-frame latent), then takes every
    ``factor``-th frame from frame 1 onwards.  This ensures each resulting
    8-frame VAE group spans ``factor`` groups of the original video.
    """
    if factor == 1:
        return list(range(num_frames))
    return [0, *list(range(1, num_frames, factor))]


def compute_latents(  # noqa: PLR0912, PLR0913, PLR0915
    dataset_file: str | Path,
    video_column: str,
    resolution_buckets: list[tuple[int, int, int]],
    output_dir: str,
    model_path: str,
    main_media_column: str | None = None,
    reshape_mode: str = "center",
    batch_size: int = 1,
    device: str = "cuda",
    vae_tiling: bool = False,
    with_audio: bool = False,
    audio_output_dir: str | None = None,
    num_dataloader_workers: int = 4,
    overwrite: bool = False,
    temporal_subsample_factor: int = 1,
    hdr_ingest: bool = False,
    hdr_transfer: str = "auto",
    hdr_synth_bracket_ev: str | None = None,
    hdr_vae_encoding: str = "reinhard",
    latent_save_dtype: torch.dtype | None = None,
    skip_existing: bool = False,
) -> None:
    """
    Process videos and save latent representations.
    Under ``accelerate launch``, each process handles an interleaved shard of
    the dataset (rank/world read from ``accelerate.PartialState``). Already-
    computed ``.pt`` outputs are skipped unless ``overwrite=True``; writes are
    atomic so an interrupted run is safe to resume.
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL) containing video paths
        video_column: Column name for video paths in the metadata file
        resolution_buckets: List of (frames, height, width) tuples
        output_dir: Directory to save video latents
        model_path: Path to LTX-2 checkpoint (.safetensors)
        reshape_mode: How to crop videos ("center", "random")
        main_media_column: Column name for main media paths (if different from video_column)
        batch_size: Batch size for processing
        device: Device to use for computation
        vae_tiling: Whether to enable VAE tiling
        with_audio: Whether to extract and encode audio from videos
        audio_output_dir: Directory to save audio latents (required if with_audio=True)
        num_dataloader_workers: Number of DataLoader worker processes (0 for in-process loading)
        overwrite: Re-process every item even if its output exists. Use when rerunning with
            changed parameters (different model, resolution, etc.) so stale outputs are replaced.
        temporal_subsample_factor: Factor for VAE-aligned temporal subsampling of reference videos
        hdr_ingest: HDR scene-linear ingest; saves ``hdr_latent`` per clip when True
        hdr_transfer: ``auto``, ``pq``, ``hlg``, ``srgb``, or ``linear`` for HDR decode
        hdr_synth_bracket_ev: Optional ``"ev_min:ev_max:step"`` for ``hdr_ldr_ev_stack`` (requires hdr_ingest)
        hdr_vae_encoding: ``reinhard``, ``pu21``, ``logc3``, or ``lf_log1p`` when hdr_ingest is True
        skip_existing: Skip shards whose output ``.pt`` already exists (maps to ``overwrite=False``)
    """
    if skip_existing:
        overwrite = False
    if latent_save_dtype is not None:
        logger.debug("latent_save_dtype=%s ignored (kino encode uses float32 latents)", latent_save_dtype)

    if hdr_synth_bracket_ev and not hdr_ingest:
        raise ValueError("hdr_synth_bracket_ev requires hdr_ingest=True")

    hve = hdr_vae_encoding.lower().strip()
    if hve not in _ALLOWED_HDR_VAE_ENCODING:
        raise ValueError(
            f"Unknown hdr_vae_encoding {hdr_vae_encoding!r}; expected one of: "
            f"{', '.join(sorted(_ALLOWED_HDR_VAE_ENCODING))}"
        )
    if hve != "reinhard" and not hdr_ingest:
        raise ValueError("hdr_vae_encoding other than 'reinhard' requires hdr_ingest=True (pu21, logc3, or lf_log1p)")

    _ev_list_for_save: list[float] | None = None
    if hdr_synth_bracket_ev:
        emin, emax, estep = parse_ev_bracket_spec(hdr_synth_bracket_ev)
        _ev_list_for_save = ev_list_arange(emin, emax, estep)
    # Validate temporal subsampling compatibility with resolution buckets
    if temporal_subsample_factor > 1:
        for frames, _h, _w in resolution_buckets:
            pixel_frames_minus_one = frames - 1
            if pixel_frames_minus_one % temporal_subsample_factor != 0:
                raise ValueError(
                    f"Frame count {frames} is not compatible with "
                    f"temporal_subsample_factor={temporal_subsample_factor}. "
                    f"(frames - 1) must be divisible by the factor."
                )
            subsampled = 1 + pixel_frames_minus_one // temporal_subsample_factor
            if (subsampled - 1) % VAE_TEMPORAL_FACTOR != 0:
                raise ValueError(
                    f"After temporal subsampling {frames} → {subsampled} frames, "
                    f"result does not satisfy (frames - 1) % {VAE_TEMPORAL_FACTOR} == 0."
                )

    if with_audio and audio_output_dir is None:
        raise ValueError("audio_output_dir must be provided when with_audio=True")

    console = Console()
    torch_device = torch.device(device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_output_path: Path | None = None
    if with_audio:
        audio_output_path = Path(audio_output_dir)
        audio_output_path.mkdir(parents=True, exist_ok=True)

    env_workers = os.environ.get("GOPEX_VAE_DATALOADER_WORKERS", "").strip()
    if env_workers:
        num_dataloader_workers = int(env_workers)
    elif with_audio and num_dataloader_workers > 0:
        num_dataloader_workers = 0

    existing_video = _collect_existing_shards(output_path)
    existing_audio = _collect_existing_shards(audio_output_path) if audio_output_path else set()
    shard_filter = skip_existing and not overwrite

    dataset = MediaDataset(
        dataset_file=dataset_file,
        main_media_column=main_media_column or video_column,
        video_column=video_column,
        resolution_buckets=resolution_buckets,
        reshape_mode=reshape_mode,
        with_audio=with_audio,
        temporal_subsample_factor=temporal_subsample_factor,
        existing_video_shards=existing_video if shard_filter else None,
        existing_audio_shards=existing_audio if shard_filter else None,
        filter_existing_shards=shard_filter,
        hdr_ingest=hdr_ingest,
        hdr_transfer=hdr_transfer,
        hdr_synth_bracket_ev=hdr_synth_bracket_ev,
        hdr_vae_encoding=hve,
    )
    logger.info(f"Loaded {len(dataset)} valid media files")

    if hdr_ingest:
        if hve == "pu21":
            logger.info(
                "HDR ingest enabled: scene-linear float32 ``hdr_latent`` in each .pt; VAE input uses X2HDR PU21."
            )
        elif hve == "logc3":
            logger.info(
                "HDR ingest enabled: scene-linear float32 ``hdr_latent`` in each .pt; VAE input uses LumiVid LogC3."
            )
        elif hve == "lf_log1p":
            logger.info(
                "HDR ingest enabled: scene-linear float32 ``hdr_latent`` in each .pt; VAE input uses LF-Diff tonemap."
            )
        else:
            logger.info(
                "HDR ingest enabled: scene-linear float32 ``hdr_latent`` in each .pt; VAE input uses Reinhard tone-map."
            )
        if hdr_synth_bracket_ev:
            logger.info(
                f"Synthetic γ-encoded LDR EV stack enabled ({hdr_synth_bracket_ev}); see ``hdr_ldr_ev_stack`` in each .pt."
            )

    # Audio processing requires batch_size=1; must be applied before the dataloader is built.
    if with_audio and batch_size > 1:
        logger.warning("Audio processing requires batch_size=1. Overriding batch_size to 1.")
        batch_size = 1

    def _is_done(idx: int) -> bool:
        return _latent_shard_done(
            dataset.main_media_relpaths[idx],
            existing_video=existing_video,
            existing_audio=existing_audio,
            require_audio=with_audio,
        )

    if skip_existing and not overwrite:
        skipped = sum(1 for i in range(len(dataset)) if _is_done(i))
        remaining = len(dataset) - skipped
        logger.info(
            f"skip_existing prefilter: {skipped:,} latent shards already on disk; "
            f"{remaining:,} media files remain"
        )

    dataloader = _build_sharded_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
        is_done=_is_done,
        overwrite=overwrite,
    )
    if dataloader is None:
        return

    with console.status(f"[bold]Loading video VAE encoder from [cyan]{model_path}[/]...", spinner="dots"):
        vae = load_video_vae_encoder(model_path, device=torch_device, dtype=torch.bfloat16)

    audio_vae_encoder = None
    audio_processor = None
    if with_audio:
        with console.status(f"[bold]Loading audio VAE encoder from [cyan]{model_path}[/]...", spinner="dots"):
            audio_vae_encoder = load_audio_vae_encoder(
                checkpoint_path=model_path,
                device=torch_device,
                dtype=torch.float32,  # Audio VAE needs float32 for quality. TODO: re-test with bfloat16.
            )
            audio_processor = AudioProcessor(
                target_sample_rate=audio_vae_encoder.sample_rate,
                mel_bins=audio_vae_encoder.mel_bins,
                mel_hop_length=audio_vae_encoder.mel_hop_length,
                n_fft=audio_vae_encoder.n_fft,
            ).to(torch_device)

    # Track audio statistics
    audio_success_count = 0
    audio_skip_count = 0

    # Process batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing videos", total=len(dataloader))

        for batch in dataloader:
            batch_size_now = len(batch["relative_path"])
            pending_rel: list[Path] = []
            skip_batch = skip_existing and batch_size_now > 0 and not overwrite
            for i in range(batch_size_now):
                output_rel_path = Path(batch["main_media_relative_path"][i]).with_suffix(".pt")
                pending_rel.append(output_rel_path)
                video_file = _resolve_existing_shard(output_path, output_rel_path, existing_video)
                audio_file = (
                    _resolve_existing_shard(audio_output_path, output_rel_path, existing_audio)
                    if audio_output_path is not None
                    else None
                )
                video_ok = video_file is not None
                audio_ok = audio_output_path is None or audio_file is not None
                if not (skip_existing and video_ok and audio_ok):
                    skip_batch = False
            if skip_batch:
                progress.advance(task)
                continue

            # Get video tensor - shape is [B, F, C, H, W] from DataLoader
            video = batch["video"]

            need_video_encode = True
            if skip_existing and batch_size_now == 1 and not overwrite:
                video_file = _resolve_existing_shard(output_path, pending_rel[0], existing_video)
                audio_file = (
                    _resolve_existing_shard(audio_output_path, pending_rel[0], existing_audio)
                    if audio_output_path is not None
                    else None
                )
                if video_file is not None and (
                    audio_output_path is None or audio_file is not None
                ):
                    progress.advance(task)
                    continue
                need_video_encode = video_file is None

            video_latent_data = None
            if need_video_encode:
                with torch.inference_mode():
                    video_latent_data = _encode_video(vae=vae, video=video, use_tiling=vae_tiling)

            # Save latents for each item in batch
            for i in range(batch_size_now):
                output_rel_path = pending_rel[i]
                output_file = _canonical_shard_path(output_path, output_rel_path)
                video_file = _resolve_existing_shard(output_path, output_rel_path, existing_video)

                # Create output directory maintaining structure
                output_file.parent.mkdir(parents=True, exist_ok=True)

                if video_latent_data is not None:
                    # Store the latent's effective fps (= source_fps / subsample factor).
                    # Downstream position math expects the rate the saved latents actually have.
                    effective_fps = batch["video_metadata"]["fps"][i].item() / temporal_subsample_factor
                    latent_data = {
                        "latents": video_latent_data["latents"][i].cpu().contiguous(),  # [C, F', H', W']
                        "num_frames": video_latent_data["num_frames"],
                        "height": video_latent_data["height"],
                        "width": video_latent_data["width"],
                        "fps": effective_fps,
                    }
                    if batch.get("hdr_latent") is not None and batch.get("hdr_meta_u8") is not None:
                        latent_data["hdr_latent"] = batch["hdr_latent"][i].cpu().contiguous()
                        latent_data["hdr_meta"] = padded_u8_to_hdr_meta_dict(batch["hdr_meta_u8"][i].cpu())
                        if batch.get("hdr_ldr_ev_stack") is not None and _ev_list_for_save is not None:
                            latent_data["hdr_ldr_ev_stack"] = batch["hdr_ldr_ev_stack"][i].cpu().contiguous()
                            latent_data["hdr_ev_list"] = _ev_list_for_save

                    _atomic_save(latent_data, output_file)
                    existing_video.add(_shard_key(output_rel_path))
                    existing_video.update(_shard_alias_keys(output_rel_path))

                # Process audio if enabled (audio is already extracted by the dataset)
                if with_audio:
                    audio_output_file = _canonical_shard_path(audio_output_path, output_rel_path)
                    audio_file = _resolve_existing_shard(
                        audio_output_path, output_rel_path, existing_audio
                    )
                    if audio_file is not None:
                        continue
                    audio_batch = batch.get("audio")
                    if audio_batch is not None:
                        # Extract the i-th item from batched audio data
                        # DataLoader collates [channels, samples] -> [batch, channels, samples]
                        audio_data = Audio(
                            waveform=audio_batch["waveform"][i],
                            sampling_rate=audio_batch["sample_rate"][i].item(),
                        )

                        # Encode audio
                        with torch.inference_mode():
                            audio_latents = _encode_audio(audio_vae_encoder, audio_processor, audio_data)

                        # Save audio latents
                        audio_output_file.parent.mkdir(parents=True, exist_ok=True)

                        audio_save_data = {
                            "latents": audio_latents["latents"].cpu().contiguous(),
                            "num_time_steps": audio_latents["num_time_steps"],
                            "frequency_bins": audio_latents["frequency_bins"],
                            "duration": audio_latents["duration"],
                        }

                        _atomic_save(audio_save_data, audio_output_file)
                        existing_audio.add(_shard_key(output_rel_path))
                        existing_audio.update(_shard_alias_keys(output_rel_path))
                        audio_success_count += 1
                    else:
                        # Video has no audio track
                        audio_skip_count += 1

            progress.advance(task)

    logger.info(f"Processed {len(dataloader.dataset)} videos -> {output_path}")  # type: ignore[arg-type]
    if with_audio:
        logger.info(
            f"Audio processing: {audio_success_count} videos with audio, "
            f"{audio_skip_count} videos without audio (skipped)"
        )


def _encode_video(
    vae: torch.nn.Module,
    video: torch.Tensor,
    dtype: torch.dtype | None = None,
    use_tiling: bool = False,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> dict[str, torch.Tensor | int]:
    """Encode video into non-patchified latent representation.
    Args:
        vae: Video VAE encoder model
        video: Input tensor of shape [B, C, F, H, W] (batch, channels, frames, height, width)
               This is the format expected by the VAE encoder.
        dtype: Target dtype for output latents
        use_tiling: Whether to use spatial tiling for memory efficiency
        tile_size: Tile size in pixels (must be divisible by 32)
        tile_overlap: Overlap between tiles in pixels (must be divisible by 32)
    Returns:
        Dict containing non-patchified latents and shape information:
        {
            "latents": Tensor[B, C, F', H', W'],  # Non-patchified format with batch dim
            "num_frames": int,  # Latent frame count
            "height": int,  # Latent height
            "width": int,  # Latent width
        }
    """
    device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    # Add batch dimension if needed
    if video.ndim == 4:
        video = video.unsqueeze(0)  # [C, F, H, W] -> [B, C, F, H, W]

    video = video.to(device=device, dtype=vae_dtype)

    # Choose encoding method based on tiling flag
    if use_tiling:
        latents = _tiled_encode_video(
            vae=vae,
            video=video,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
    else:
        # Encode video - VAE expects [B, C, F, H, W], returns [B, C, F', H', W']
        latents = vae(video)

    if dtype is not None:
        latents = latents.to(dtype=dtype)

    _, _, num_frames, height, width = latents.shape

    return {
        "latents": latents,  # [B, C, F', H', W']
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }


def _tiled_encode_video(  # noqa: PLR0912, PLR0915
    vae: torch.nn.Module,
    video: torch.Tensor,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> torch.Tensor:
    """Encode video using spatial tiling for memory efficiency.
    Splits the video into overlapping spatial tiles, encodes each tile separately,
    and blends the results using linear feathering in the overlap regions.
    Args:
        vae: Video VAE encoder model
        video: Input tensor of shape [B, C, F, H, W]
        tile_size: Tile size in pixels (must be divisible by 32)
        tile_overlap: Overlap between tiles in pixels (must be divisible by 32)
    Returns:
        Encoded latent tensor [B, C_latent, F_latent, H_latent, W_latent]
    """
    batch, _channels, frames, height, width = video.shape
    device = video.device
    dtype = video.dtype

    # Validate tile parameters
    if tile_size % VAE_SPATIAL_FACTOR != 0:
        raise ValueError(f"tile_size must be divisible by {VAE_SPATIAL_FACTOR}, got {tile_size}")
    if tile_overlap % VAE_SPATIAL_FACTOR != 0:
        raise ValueError(f"tile_overlap must be divisible by {VAE_SPATIAL_FACTOR}, got {tile_overlap}")
    if tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap ({tile_overlap}) must be less than tile_size ({tile_size})")

    # If video fits in a single tile, use regular encoding
    if height <= tile_size and width <= tile_size:
        return vae(video)

    # Calculate output dimensions
    # VAE compresses: H -> H/32, W -> W/32, F -> 1 + (F-1)/8
    output_height = height // VAE_SPATIAL_FACTOR
    output_width = width // VAE_SPATIAL_FACTOR
    output_frames = 1 + (frames - 1) // VAE_TEMPORAL_FACTOR

    # Latent channels (128 for LTX-2)
    # Get from a small test encode or assume 128
    latent_channels = 128

    # Initialize output and weight tensors
    output = torch.zeros(
        (batch, latent_channels, output_frames, output_height, output_width),
        device=device,
        dtype=dtype,
    )
    weights = torch.zeros(
        (batch, 1, output_frames, output_height, output_width),
        device=device,
        dtype=dtype,
    )

    # Calculate tile positions with overlap
    # Step size is tile_size - tile_overlap
    step_h = tile_size - tile_overlap
    step_w = tile_size - tile_overlap

    h_positions = list(range(0, max(1, height - tile_overlap), step_h))
    w_positions = list(range(0, max(1, width - tile_overlap), step_w))

    # Ensure last tile covers the edge
    if h_positions[-1] + tile_size < height:
        h_positions.append(height - tile_size)
    if w_positions[-1] + tile_size < width:
        w_positions.append(width - tile_size)

    # Remove duplicates and sort
    h_positions = sorted(set(h_positions))
    w_positions = sorted(set(w_positions))

    # Overlap in latent space
    overlap_out_h = tile_overlap // VAE_SPATIAL_FACTOR
    overlap_out_w = tile_overlap // VAE_SPATIAL_FACTOR

    # Process each tile
    for h_pos in h_positions:
        for w_pos in w_positions:
            # Calculate tile boundaries in input space
            h_start = max(0, h_pos)
            w_start = max(0, w_pos)
            h_end = min(h_start + tile_size, height)
            w_end = min(w_start + tile_size, width)

            # Ensure tile dimensions are divisible by VAE_SPATIAL_FACTOR
            tile_h = ((h_end - h_start) // VAE_SPATIAL_FACTOR) * VAE_SPATIAL_FACTOR
            tile_w = ((w_end - w_start) // VAE_SPATIAL_FACTOR) * VAE_SPATIAL_FACTOR

            if tile_h < VAE_SPATIAL_FACTOR or tile_w < VAE_SPATIAL_FACTOR:
                continue

            # Adjust end positions
            h_end = h_start + tile_h
            w_end = w_start + tile_w

            # Extract tile
            tile = video[:, :, :, h_start:h_end, w_start:w_end]

            # Encode tile
            encoded_tile = vae(tile)

            # Get actual encoded dimensions
            _, _, tile_out_frames, tile_out_height, tile_out_width = encoded_tile.shape

            # Calculate output positions
            out_h_start = h_start // VAE_SPATIAL_FACTOR
            out_w_start = w_start // VAE_SPATIAL_FACTOR
            out_h_end = min(out_h_start + tile_out_height, output_height)
            out_w_end = min(out_w_start + tile_out_width, output_width)

            # Trim encoded tile if necessary
            actual_tile_h = out_h_end - out_h_start
            actual_tile_w = out_w_end - out_w_start
            encoded_tile = encoded_tile[:, :, :, :actual_tile_h, :actual_tile_w]

            # Create blending mask with linear feathering at edges
            mask = torch.ones(
                (1, 1, tile_out_frames, actual_tile_h, actual_tile_w),
                device=device,
                dtype=dtype,
            )

            # Apply feathering at edges (linear blend in overlap regions)
            # Left edge
            if h_pos > 0 and overlap_out_h > 0 and overlap_out_h < actual_tile_h:
                fade_in = torch.linspace(0.0, 1.0, overlap_out_h + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, :overlap_out_h, :] *= fade_in.view(1, 1, 1, -1, 1)

            # Right edge (bottom in height dimension)
            if h_end < height and overlap_out_h > 0 and overlap_out_h < actual_tile_h:
                fade_out = torch.linspace(1.0, 0.0, overlap_out_h + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, -overlap_out_h:, :] *= fade_out.view(1, 1, 1, -1, 1)

            # Top edge (left in width dimension)
            if w_pos > 0 and overlap_out_w > 0 and overlap_out_w < actual_tile_w:
                fade_in = torch.linspace(0.0, 1.0, overlap_out_w + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, :, :overlap_out_w] *= fade_in.view(1, 1, 1, 1, -1)

            # Bottom edge (right in width dimension)
            if w_end < width and overlap_out_w > 0 and overlap_out_w < actual_tile_w:
                fade_out = torch.linspace(1.0, 0.0, overlap_out_w + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, :, -overlap_out_w:] *= fade_out.view(1, 1, 1, 1, -1)

            # Accumulate weighted results
            output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += encoded_tile * mask
            weights[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += mask

    # Normalize by weights (avoid division by zero)
    output = output / (weights + 1e-8)

    return output


def _encode_audio(
    audio_vae_encoder: torch.nn.Module,
    audio_processor: torch.nn.Module,
    audio: Audio,
) -> dict[str, torch.Tensor | int | float]:
    """Encode audio waveform into latent representation.
    Args:
        audio_vae_encoder: Audio VAE encoder model from ltx-core
        audio_processor: AudioProcessor for waveform-to-spectrogram conversion
        audio: Audio container with waveform tensor and sampling rate.
    Returns:
        Dict containing audio latents and shape information:
        {
            "latents": Tensor[C, T, F],  # Non-patchified format
            "num_time_steps": int,
            "frequency_bins": int,
            "duration": float,
        }
    """
    device = next(audio_vae_encoder.parameters()).device
    dtype = next(audio_vae_encoder.parameters()).dtype

    waveform = audio.waveform.to(device=device, dtype=dtype)

    # Add batch dimension if needed: [channels, samples] -> [batch, channels, samples]
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    # Convert to stereo if needed (audio VAE expects 2 channels)
    # Channel order for surround: 5.1=[L,R,C,LFE,Ls,Rs], 7.1=[L,R,C,LFE,Ls,Rs,Lb,Rb]
    num_channels = waveform.shape[1]
    if num_channels == 1:
        # Mono to stereo: duplicate the channel
        waveform = waveform.repeat(1, 2, 1)
    elif num_channels == 6:
        # 5.1 downmix with normalized weights (sum to 1.0)
        # Original: L = L + 0.707*C + 0.707*Ls, weights sum = 2.414
        w_main = 1.0 / 2.414  # ~0.414
        w_other = 0.707 / 2.414  # ~0.293
        left = w_main * waveform[:, 0, :] + w_other * waveform[:, 2, :] + w_other * waveform[:, 4, :]
        right = w_main * waveform[:, 1, :] + w_other * waveform[:, 2, :] + w_other * waveform[:, 5, :]
        waveform = torch.stack([left, right], dim=1)
    elif num_channels == 8:
        # 7.1 downmix with normalized weights (sum to 1.0)
        # Original: L = L + 0.707*C + 0.707*Ls + 0.707*Lb, weights sum = 3.121
        w_main = 1.0 / 3.121  # ~0.320
        w_other = 0.707 / 3.121  # ~0.227
        center = waveform[:, 2, :]
        left = w_main * waveform[:, 0, :] + w_other * (center + waveform[:, 4, :] + waveform[:, 6, :])
        right = w_main * waveform[:, 1, :] + w_other * (center + waveform[:, 5, :] + waveform[:, 7, :])
        waveform = torch.stack([left, right], dim=1)
    elif num_channels > 2:
        # Unknown layout: average all channels to mono, then duplicate to stereo
        logger.warning(f"Unknown audio channel layout ({num_channels} channels), using mean downmix")
        mono = waveform.mean(dim=1, keepdim=True)
        waveform = mono.repeat(1, 2, 1)

    # Calculate duration
    duration = waveform.shape[-1] / audio.sampling_rate

    # Convert waveform to mel spectrogram using AudioProcessor
    mel_spectrogram = audio_processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=audio.sampling_rate))
    mel_spectrogram = mel_spectrogram.to(dtype=dtype)

    # Encode mel spectrogram to latents
    latents = audio_vae_encoder(mel_spectrogram)

    # latents shape: [batch, channels, time, freq] = [1, 8, T, 16]
    _, _channels, time_steps, freq_bins = latents.shape

    return {
        "latents": latents.squeeze(0),  # [C, T, F] - remove batch dim
        "num_time_steps": time_steps,
        "frequency_bins": freq_bins,
        "duration": duration,
    }


AUDIO_FILE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"}
VIDEO_FILE_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".bmp", ".tiff", ".webp"}


def compute_video_masks(
    dataset_file: str | Path,
    mask_column: str,
    latents_dir: str,
    output_dir: str,
    main_media_column: str | None = None,
    overwrite: bool = False,
) -> None:
    """Preprocess video mask files to latent-space binary masks.
    For each sample, loads the mask video/image, applies the same spatial
    resize/crop as the target video (read from saved latent metadata), downsamples
    to latent dimensions, binarizes, and saves as a .pt tensor.
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL).
        mask_column: Column name containing mask video/image paths.
        latents_dir: Directory containing the target video latents (for reading
            spatial/temporal metadata to ensure mask alignment).
        output_dir: Directory to save mask .pt files.
        main_media_column: Column for output file naming (defaults to mask_column).
    """
    dataset_path = Path(dataset_file)
    data_root = dataset_path.parent
    latents_path = Path(latents_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    naming_column = main_media_column or mask_column
    mask_paths = _load_paths_from_dataset(dataset_path, mask_column)
    naming_paths = _load_paths_from_dataset(dataset_path, naming_column) if naming_column != mask_column else mask_paths

    success = 0
    for mask_file, naming_file in zip(mask_paths, naming_paths, strict=True):
        rel_path = _output_relative(naming_file, data_root)
        latent_file = latents_path / rel_path.with_suffix(".pt")
        out_file = output_path / rel_path.with_suffix(".pt")

        if not latent_file.exists():
            logger.warning(f"No target latent found at {latent_file}, skipping mask {mask_file}")
            continue

        if not overwrite and out_file.is_file():
            continue

        target_meta = torch.load(latent_file, map_location="cpu", weights_only=True)
        latent_f = target_meta["num_frames"]
        latent_h = target_meta["height"]
        latent_w = target_meta["width"]
        pixel_h = latent_h * VAE_SPATIAL_FACTOR
        pixel_w = latent_w * VAE_SPATIAL_FACTOR
        pixel_f = (latent_f - 1) * VAE_TEMPORAL_FACTOR + 1

        # Load mask as video or image
        if mask_file.suffix.lower() in IMAGE_FILE_EXTENSIONS:
            img = to_tensor(open_image_as_srgb(mask_file)).mean(dim=0, keepdim=True)  # [1, H, W]
            img = tv_resize(img.unsqueeze(0), [pixel_h, pixel_w]).squeeze(0)  # [1, H, W]
            mask_pixels = img.expand(pixel_f, -1, -1)  # tile across frames → [F, H, W]
        else:
            frames, _ = read_video(str(mask_file), max_frames=pixel_f)  # [F, C, H, W]
            frames = frames[:pixel_f].mean(dim=1)  # grayscale → [F, H, W]
            frames = torch.nn.functional.interpolate(
                frames.unsqueeze(1), size=(pixel_h, pixel_w), mode="nearest"
            ).squeeze(1)  # [F, H, W]
            mask_pixels = frames

        # Downsample to latent dims: [F, H, W] → [F', H', W']
        mask_latent = torch.nn.functional.avg_pool2d(mask_pixels.unsqueeze(1), kernel_size=VAE_SPATIAL_FACTOR).squeeze(
            1
        )  # [F, H', W'] → spatial done
        # Temporal: max-pool over groups of VAE_TEMPORAL_FACTOR frames (any masked frame masks the group)
        f_spatial = mask_latent.shape[0]
        pad_f = (VAE_TEMPORAL_FACTOR - f_spatial % VAE_TEMPORAL_FACTOR) % VAE_TEMPORAL_FACTOR
        if pad_f > 0:
            mask_latent = torch.nn.functional.pad(mask_latent, (0, 0, 0, 0, 0, pad_f))
        h_prime, w_prime = mask_latent.shape[1], mask_latent.shape[2]
        mask_latent = mask_latent.reshape(-1, VAE_TEMPORAL_FACTOR, h_prime, w_prime).amax(dim=1)[:latent_f]

        # Binarize
        mask_latent = (mask_latent > 0.5).float()

        out_file.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save({"mask": mask_latent}, out_file)
        success += 1

    logger.info(f"Mask preprocessing complete: {success} masks saved to {output_path}")


def compute_audio_masks(
    dataset_file: str | Path,
    mask_column: str,
    audio_latents_dir: str,
    output_dir: str,
    main_media_column: str | None = None,
    overwrite: bool = False,
) -> None:
    """Preprocess audio mask files to latent-space binary masks.
    For each sample, loads the mask (a 1D waveform-like signal or a simple tensor),
    resamples it to match the target audio latent temporal length, binarizes, and saves.
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL).
        mask_column: Column name containing mask file paths (.wav or .pt).
        audio_latents_dir: Directory containing the target audio latents (for reading
            temporal metadata to ensure mask alignment).
        output_dir: Directory to save mask .pt files.
        main_media_column: Column for output file naming (defaults to mask_column).
    """
    dataset_path = Path(dataset_file)
    data_root = dataset_path.parent
    audio_latents_path = Path(audio_latents_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    naming_column = main_media_column or mask_column
    mask_paths = _load_paths_from_dataset(dataset_path, mask_column)
    naming_paths = _load_paths_from_dataset(dataset_path, naming_column) if naming_column != mask_column else mask_paths

    success = 0
    for mask_file, naming_file in zip(mask_paths, naming_paths, strict=True):
        rel_path = _output_relative(naming_file, data_root)
        latent_file = audio_latents_path / rel_path.with_suffix(".pt")
        out_file = output_path / rel_path.with_suffix(".pt")

        if not latent_file.exists():
            logger.warning(f"No target audio latent found at {latent_file}, skipping mask {mask_file}")
            continue

        if not overwrite and out_file.is_file():
            continue

        target_meta = torch.load(latent_file, map_location="cpu", weights_only=True)
        latent_t = target_meta["num_time_steps"]

        # Load mask: .pt file (raw tensor) or .wav (use amplitude envelope)
        if mask_file.suffix == ".pt":
            raw_mask = torch.load(mask_file, map_location="cpu", weights_only=True)
            if isinstance(raw_mask, dict):
                raw_mask = raw_mask.get("mask", next(iter(raw_mask.values())))
            raw_mask = raw_mask.float().flatten()
        else:
            audio = _load_audio_from_file(mask_file)
            if audio is None:
                logger.warning(f"Could not load audio mask from {mask_file}")
                continue
            raw_mask = audio.waveform.abs().mean(dim=0)  # mono amplitude envelope

        # Resample to target audio latent length
        mask_resampled = torch.nn.functional.interpolate(
            raw_mask.unsqueeze(0).unsqueeze(0), size=latent_t, mode="nearest"
        ).squeeze()  # [latent_t]

        mask_binary = (mask_resampled > 0.5).float()

        out_file.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save({"mask": mask_binary}, out_file)
        success += 1

    logger.info(f"Audio mask preprocessing complete: {success} masks saved to {output_path}")


def compute_audio_latents(  # noqa: PLR0915
    dataset_file: str | Path,
    audio_column: str,
    output_dir: str,
    model_path: str,
    main_media_column: str | None = None,
    max_duration: float | None = None,
    duration_buckets: list[float] | None = None,
    device: str = "cuda",
    overwrite: bool = False,
) -> None:
    """Encode audio files into latent representations.
    Supports standalone audio files (.wav, .mp3, etc.) and audio tracks
    extracted from video files (.mp4, etc.).
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL).
        audio_column: Column name containing audio file paths.
        output_dir: Directory to save audio latents.
        model_path: Path to LTX-2 checkpoint (.safetensors).
        main_media_column: Column for output file naming (defaults to audio_column).
            Ensures alignment with other latent directories.
        max_duration: Maximum audio duration in seconds. Audio is trimmed if longer.
            Mutually exclusive with duration_buckets.
        duration_buckets: List of allowed durations in seconds (e.g. [2.0, 4.0, 8.0]).
            Each audio file is matched to the largest bucket that fits its duration,
            then trimmed to exactly that length. Files shorter than the smallest
            bucket are skipped. Ensures uniform lengths for batched training.
        device: Device to use for computation.
    """
    console = Console()
    torch_device = torch.device(device)

    dataset_path = Path(dataset_file)
    data_root = dataset_path.parent
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    naming_column = main_media_column or audio_column
    audio_paths = _load_paths_from_dataset(dataset_path, audio_column)
    naming_paths = (
        _load_paths_from_dataset(dataset_path, naming_column) if naming_column != audio_column else audio_paths
    )

    with console.status(f"[bold]Loading audio VAE encoder from [cyan]{model_path}[/]...", spinner="dots"):
        audio_vae_encoder = load_audio_vae_encoder(
            checkpoint_path=model_path,
            device=torch_device,
            dtype=torch.float32,
        )
        audio_processor = AudioProcessor(
            target_sample_rate=audio_vae_encoder.sample_rate,
            mel_bins=audio_vae_encoder.mel_bins,
            mel_hop_length=audio_vae_encoder.mel_hop_length,
            n_fft=audio_vae_encoder.n_fft,
        ).to(torch_device)

    sorted_buckets = sorted(duration_buckets, reverse=True) if duration_buckets else None
    success_count = 0
    skip_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Encoding audio", total=len(audio_paths))

        for audio_path, naming_path in zip(audio_paths, naming_paths, strict=True):
            rel_path = _output_relative(naming_path, data_root)
            output_file = output_path / rel_path.with_suffix(".pt")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if not overwrite and output_file.is_file():
                success_count += 1
                progress.advance(task)
                continue

            # Load audio (no trimming yet — need full duration for bucket matching)
            audio = _load_audio_from_file(audio_path)
            if audio is None:
                skip_count += 1
                progress.advance(task)
                continue

            file_duration = audio.waveform.shape[-1] / audio.sampling_rate

            # Determine target duration: bucket matching, max_duration cap, or full file
            target_duration = file_duration
            if sorted_buckets:
                bucket = next((b for b in sorted_buckets if b <= file_duration), None)
                if bucket is None:
                    logger.warning(
                        f"Skipping {audio_path.name} ({file_duration:.1f}s) — shorter than "
                        f"smallest bucket ({sorted_buckets[-1]:.1f}s)"
                    )
                    skip_count += 1
                    progress.advance(task)
                    continue
                target_duration = bucket
            elif max_duration is not None:
                target_duration = min(file_duration, max_duration)

            # Trim to target duration
            target_samples = int(target_duration * audio.sampling_rate)
            trimmed_waveform = audio.waveform[:, :target_samples]
            audio = Audio(waveform=trimmed_waveform, sampling_rate=audio.sampling_rate)

            with torch.inference_mode():
                audio_latents = _encode_audio(audio_vae_encoder, audio_processor, audio)

            _atomic_save(
                {
                    "latents": audio_latents["latents"].cpu().contiguous(),
                    "num_time_steps": audio_latents["num_time_steps"],
                    "frequency_bins": audio_latents["frequency_bins"],
                    "duration": audio_latents["duration"],
                },
                output_file,
            )
            success_count += 1
            progress.advance(task)

    logger.info(f"Audio encoding complete: {success_count} encoded, {skip_count} skipped. Saved to {output_path}")


def _output_relative(path: Path, data_root: Path) -> Path:
    """Relative path used to name a sample's cached output, mirroring the input layout.
    Normally media lives under the dataset directory and this is just the path relative to it.
    If a media path is absolute or otherwise outside the dataset directory (e.g. a one-off
    metadata file that references media elsewhere), mirror its absolute structure under the
    output directory instead of raising, so out-of-tree media stays collision-free.
    """
    try:
        return path.relative_to(data_root)
    except ValueError:
        return Path(*path.parts[1:]) if path.is_absolute() else path


def _resolve_dataset_media_path(dataset_file: Path, rel: str) -> Path:
    """Resolve ``media_path`` from a manifest row.

    Gopex manifests under ``<root>/ltx_manifest/dataset.json`` store paths like
    ``ltx_manifest/clips/foo.mp4`` or ``data/foo.png`` relative to ``<root>``.
    Legacy ltx-trainer manifests use paths relative to the manifest file directory.
    """
    p = Path(rel.strip())
    manifest_dir = dataset_file.parent
    if manifest_dir.name == "ltx_manifest":
        dataset_root = manifest_dir.parent
        if p.parts[:1] == ("ltx_manifest",):
            return dataset_root / p
        rooted = dataset_root / p
        legacy = manifest_dir / p
        if legacy.is_file() and not rooted.is_file():
            return legacy
        return rooted
    if p.parts[:1] == ("ltx_manifest",):
        return manifest_dir / p
    return manifest_dir / p


resolve_dataset_media_path = _resolve_dataset_media_path


def _manifest_media_relpath(media_rel: str) -> Path:
    """Manifest ``media_path`` value as a Path (keeps ``ltx_manifest/`` prefix when present)."""
    return Path(media_rel.strip())


def _shard_key(rel_pt: Path) -> str:
    return rel_pt.as_posix()


def _shard_alias_keys(rel_pt: Path) -> set[str]:
    """Legacy layouts: ``clips/foo.pt`` vs ``ltx_manifest/clips/foo.pt``."""
    parts = rel_pt.parts
    aliases: set[str] = set()
    if parts[:1] == ("ltx_manifest",) and len(parts) > 1:
        aliases.add(Path("clips", *parts[2:]).as_posix())
    elif parts[:1] == ("clips",):
        aliases.add(Path("ltx_manifest", *parts).as_posix())
    return aliases


def _canonical_shard_path(root: Path | None, rel_pt: Path) -> Path:
    if root is None:
        raise ValueError("output root is required")
    return root / rel_pt


def _collect_existing_shards(root: Path | None) -> set[str]:
    """One-time scan of ``*.pt`` shards under ``root`` (includes legacy alias keys)."""
    if root is None or not root.is_dir():
        return set()
    found: set[str] = set()
    for path in root.rglob("*.pt"):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        found.add(rel.as_posix())
        found.update(_shard_alias_keys(rel))
    return found


def _resolve_existing_shard(
    root: Path | None,
    rel_pt: Path,
    existing: set[str],
) -> Path | None:
    if root is None:
        return None
    for key in (_shard_key(rel_pt), *_shard_alias_keys(rel_pt)):
        if key in existing:
            candidate = root / key
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None


def _latent_shard_done(
    media_relpath: Path,
    *,
    existing_video: set[str],
    existing_audio: set[str],
    require_audio: bool,
) -> bool:
    rel_pt = media_relpath.with_suffix(".pt")
    keys = {_shard_key(rel_pt), *_shard_alias_keys(rel_pt)}
    if not (keys & existing_video):
        return False
    if not require_audio:
        return True
    return bool(keys & existing_audio)


def _load_media_column_from_dataset(dataset_file: Path, column: str) -> tuple[list[Path], list[Path]]:
    """Return resolved media paths and manifest-relative paths from a dataset column."""

    def _rows_from_csv() -> list[dict[str, str]]:
        df = pd.read_csv(dataset_file)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV file")
        return [{column: str(v)} for v in df[column].tolist()]

    def _rows_from_json() -> list[dict[str, str]]:
        with open(dataset_file, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        return data

    def _rows_from_jsonl() -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with open(dataset_file, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    if dataset_file.suffix == ".csv":
        rows = _rows_from_csv()
    elif dataset_file.suffix == ".json":
        rows = _rows_from_json()
    elif dataset_file.suffix == ".jsonl":
        rows = _rows_from_jsonl()
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_file.suffix}")

    paths: list[Path] = []
    relpaths: list[Path] = []
    for entry in rows:
        raw = str(entry[column])
        paths.append(_resolve_dataset_media_path(dataset_file, raw))
        relpaths.append(_manifest_media_relpath(raw))
    return paths, relpaths


def _load_paths_from_dataset(dataset_file: Path, column: str) -> list[Path]:
    """Load file paths from a dataset column."""
    paths, _relpaths = _load_media_column_from_dataset(dataset_file, column)
    return paths


def _load_audio_from_file(audio_path: Path, max_duration: float | None = None) -> Audio | None:
    """Load audio from an audio or video file, optionally trimming to max_duration."""
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
    except Exception:
        if os.environ.get("GOPEX_LOG_MISSING_AUDIO", "0").strip().lower() in ("1", "true", "yes", "on"):
            logger.debug(f"Could not load audio from {audio_path}")
        return None

    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:, :max_samples]

    return Audio(waveform=waveform, sampling_rate=sample_rate)


def detect_dataset_columns(dataset_file: str | Path) -> set[str]:
    """Read column names from a dataset file without loading all data."""
    path = Path(dataset_file)
    if path.suffix == ".csv":
        df = pd.read_csv(path, nrows=0)
        return set(df.columns)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data[0].keys()) if isinstance(data, list) and data else set()
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return set(json.loads(f.readline()).keys())
    return set()


def parse_resolution_buckets(resolution_buckets_str: str) -> list[tuple[int, int, int]]:
    """Parse resolution buckets from string format to list of tuples (frames, height, width)"""
    resolution_buckets = []
    for bucket_str in resolution_buckets_str.split(";"):
        w, h, f = map(int, bucket_str.split("x"))

        if w % VAE_SPATIAL_FACTOR != 0 or h % VAE_SPATIAL_FACTOR != 0:
            raise typer.BadParameter(
                f"Width and height must be multiples of {VAE_SPATIAL_FACTOR}, got {w}x{h}",
                param_hint="resolution-buckets",
            )

        if f % VAE_TEMPORAL_FACTOR != 1:
            raise typer.BadParameter(
                f"Number of frames must be a multiple of {VAE_TEMPORAL_FACTOR} plus 1, got {f}",
                param_hint="resolution-buckets",
            )

        resolution_buckets.append((f, h, w))
    return resolution_buckets


def compute_scaled_resolution_buckets(
    resolution_buckets: list[tuple[int, int, int]],
    scale_factor: int,
) -> list[tuple[int, int, int]]:
    """Compute scaled resolution buckets and validate the results."""
    if scale_factor == 1:
        return resolution_buckets

    scaled_buckets = []
    for frames, height, width in resolution_buckets:
        # Validate that scale factor evenly divides the dimensions
        if height % scale_factor != 0:
            raise ValueError(
                f"Height {height} is not evenly divisible by scale factor {scale_factor}. "
                f"Choose a scale factor that divides {height} evenly."
            )
        if width % scale_factor != 0:
            raise ValueError(
                f"Width {width} is not evenly divisible by scale factor {scale_factor}. "
                f"Choose a scale factor that divides {width} evenly."
            )

        scaled_height = height // scale_factor
        scaled_width = width // scale_factor

        # Validate scaled dimensions are divisible by VAE spatial factor
        if scaled_height % VAE_SPATIAL_FACTOR != 0:
            raise ValueError(
                f"Scaled height {scaled_height} (from {height} / {scale_factor}) "
                f"is not divisible by {VAE_SPATIAL_FACTOR}. "
                f"Choose a different scale factor or adjust your resolution buckets."
            )
        if scaled_width % VAE_SPATIAL_FACTOR != 0:
            raise ValueError(
                f"Scaled width {scaled_width} (from {width} / {scale_factor}) "
                f"is not divisible by {VAE_SPATIAL_FACTOR}. "
                f"Choose a different scale factor or adjust your resolution buckets."
            )

        scaled_buckets.append((frames, scaled_height, scaled_width))

    return scaled_buckets


def _atomic_save(data: Any, out: Path) -> None:  # noqa: ANN401
    """Save to ``out`` atomically via per-PID temp file + replace.
    Crash mid-write leaves an orphan ``.tmp.<pid>`` file that the skip logic
    ignores. The per-PID suffix makes concurrent writes from multiple ranks
    collision-free.
    """
    tmp = out.with_suffix(f"{out.suffix}.tmp.{os.getpid()}")
    torch.save(data, tmp)
    tmp.replace(out)


def _build_sharded_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    is_done: Callable[[int], bool],
    overwrite: bool,
) -> DataLoader | None:
    """Return a DataLoader over this rank's interleaved shard of ``dataset``.
    When ``overwrite`` is False, items whose outputs already exist (per
    ``is_done``) are filtered out. Returns ``None`` if this rank has nothing
    to do, so the caller can early-return without loading any models.
    """
    state = PartialState()
    todo = [i for i in range(state.process_index, len(dataset), state.num_processes) if overwrite or not is_done(i)]
    if not todo:
        logger.info(f"Rank {state.process_index}/{state.num_processes}: nothing to do")
        return None
    logger.info(f"Rank {state.process_index}/{state.num_processes}: processing {len(todo):,} of {len(dataset):,} items")
    return DataLoader(Subset(dataset, todo), batch_size=batch_size, shuffle=False, num_workers=num_workers)


@app.command()
def main(  # noqa: PLR0913
    dataset_file: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing video paths",
    ),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF;WxHxF;..." (e.g. "768x768x25;512x512x49")',
    ),
    output_dir: str = typer.Option(
        ...,
        help="Output directory to save video latents",
    ),
    model_path: str = typer.Option(
        ...,
        help="Path to LTX-2 checkpoint (.safetensors file)",
    ),
    video_column: str = typer.Option(
        default="media_path",
        help="Column name in the dataset JSON/JSONL/CSV file containing video paths",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size for processing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    reshape_mode: str = typer.Option(
        default="center",
        help="How to crop videos: 'center' or 'random'",
    ),
    with_audio: bool = typer.Option(
        default=False,
        help="Extract and encode audio from video files",
    ),
    audio_output_dir: str | None = typer.Option(
        default=None,
        help="Output directory for audio latents (required if --with-audio is set)",
    ),
    overwrite: bool = typer.Option(
        default=False,
        help="Re-encode every item even if its output exists. Use when rerunning with "
        "changed parameters (different model, resolution, etc.) so stale outputs are replaced.",
    ),
    hdr_ingest: bool = typer.Option(
        default=False,
        help="HDR decode to scene-linear float32; save hdr_latent in each .pt; tone-map for VAE input",
    ),
    hdr_transfer: str = typer.Option(
        default="auto",
        help="Color transfer for HDR linearization: auto, pq, hlg, srgb, or linear",
    ),
    hdr_synth_bracket_ev: str | None = typer.Option(
        default=None,
        help='Optional LatentHDR-style synthetic γ-LDR stack: "ev_min:ev_max:step" (e.g. "-7:5:1"); requires --hdr-ingest',
    ),
    hdr_vae_encoding: str = typer.Option(
        default="reinhard",
        help="With --hdr-ingest: VAE pixel encoding: reinhard | pu21 | logc3 | lf_log1p",
    ),
) -> None:
    """Process videos/images and save latent representations for video generation training.
    This script processes videos and images from metadata files and saves latent representations
    that can be used for training video generation models. The output latents will maintain
    the same folder structure and naming as the corresponding media files.
    For multi-GPU preprocessing, invoke under ``accelerate launch`` -- each process
    will handle an interleaved shard of the dataset.
    Examples:
        # Process videos from a CSV file
        python scripts/process_videos.py dataset.csv --resolution-buckets 768x768x25 \\
            --output-dir ./latents --model-path /path/to/ltx2.safetensors
        # Process videos from a JSON file with custom video column
        python scripts/process_videos.py dataset.json --resolution-buckets 768x768x25 \\
            --output-dir ./latents --model-path /path/to/ltx2.safetensors --video-column "video_path"
        # Enable VAE tiling to save GPU VRAM
        python scripts/process_videos.py dataset.csv --resolution-buckets 1024x1024x25 \\
            --output-dir ./latents --model-path /path/to/ltx2.safetensors --vae-tiling
        # Process videos with audio
        python scripts/process_videos.py dataset.csv --resolution-buckets 768x768x25 \\
            --output-dir ./latents --model-path /path/to/ltx2.safetensors \\
            --with-audio --audio-output-dir ./audio_latents
    """

    # Validate dataset file exists
    if not Path(dataset_file).is_file():
        raise typer.BadParameter(f"Dataset file not found: {dataset_file}")

    # Validate audio parameters
    if with_audio and audio_output_dir is None:
        raise typer.BadParameter("--audio-output-dir is required when --with-audio is set")

    ht = hdr_transfer.lower().strip()
    if ht not in _ALLOWED_HDR_TRANSFER:
        raise typer.BadParameter(
            f"Unknown hdr-transfer {hdr_transfer!r}; expected one of: {', '.join(sorted(_ALLOWED_HDR_TRANSFER))}"
        )
    if hdr_synth_bracket_ev and not hdr_ingest:
        raise typer.BadParameter("--hdr-synth-bracket-ev requires --hdr-ingest")
    hve = hdr_vae_encoding.lower().strip()
    if hve not in _ALLOWED_HDR_VAE_ENCODING:
        raise typer.BadParameter(
            f"Unknown --hdr-vae-encoding {hdr_vae_encoding!r}; expected one of: "
            f"{', '.join(sorted(_ALLOWED_HDR_VAE_ENCODING))}"
        )
    if hve != "reinhard" and not hdr_ingest:
        raise typer.BadParameter("--hdr-vae-encoding other than reinhard requires --hdr-ingest")

    # Parse resolution buckets
    parsed_resolution_buckets = parse_resolution_buckets(resolution_buckets)

    if len(parsed_resolution_buckets) > 1:
        logger.warning(
            "Using multiple resolution buckets. "
            "When training with multiple resolution buckets, you must use a batch size of 1."
        )

    # Process latents
    compute_latents(
        dataset_file=dataset_file,
        video_column=video_column,
        resolution_buckets=parsed_resolution_buckets,
        output_dir=output_dir,
        model_path=model_path,
        reshape_mode=reshape_mode,
        batch_size=batch_size,
        device=device,
        vae_tiling=vae_tiling,
        with_audio=with_audio,
        audio_output_dir=audio_output_dir,
        overwrite=overwrite,
        hdr_ingest=hdr_ingest,
        hdr_transfer=ht,
        hdr_synth_bracket_ev=hdr_synth_bracket_ev,
        hdr_vae_encoding=hve,
    )


if __name__ == "__main__":
    app()
