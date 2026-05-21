# ruff: noqa: PLC0415

"""
Model loader for LTX-2 trainer using the new ltx-core package.
This module provides a unified interface for loading LTX-2 model components
for training, using SingleGPUModelBuilder from ltx-core.
Example usage:
    # Load individual components
    vae_encoder = load_video_vae_encoder("/path/to/checkpoint.safetensors", device="cuda")
    vae_decoder = load_video_vae_decoder("/path/to/checkpoint.safetensors", device="cuda")
    text_encoder = load_text_encoder("/path/to/gemma", device="cuda")
    # Load all components at once
    components = load_model("/path/to/checkpoint.safetensors", text_encoder_path="/path/to/gemma")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ltx_trainer import logger
from ltx_trainer.nvml_safe_cuda import apply_nvml_safe_cuda_patches

apply_nvml_safe_cuda_patches()

import torch

# Type alias for device specification
Device = str | torch.device

# Type checking imports (not loaded at runtime)
if TYPE_CHECKING:
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.audio_vae import AudioDecoder, AudioEncoder, Vocoder
    from ltx_core.model.transformer import LTXModel
    from ltx_core.model.video_vae import VideoDecoder, VideoEncoder
    from ltx_core.text_encoders.gemma import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor


def _to_torch_device(device: Device) -> torch.device:
    """Convert device specification to torch.device."""
    return torch.device(device) if isinstance(device, str) else device


# =============================================================================
# Individual Component Loaders
# =============================================================================


def load_transformer(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "LTXModel":
    """Load the LTX transformer model.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded LTXModel transformer
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoEncoder":
    """Load the video VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoEncoder
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import VAE_ENCODER_COMFY_KEYS_FILTER, VideoEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoDecoder":
    """Load the video VAE decoder (for inference/validation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoDecoder
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import VAE_DECODER_COMFY_KEYS_FILTER, VideoDecoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioEncoder":
    """Load the audio VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights (default bfloat16, but float32 recommended for quality)
    Returns:
        Loaded AudioEncoder
    """
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioDecoder":
    """Load the audio VAE decoder.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded AudioDecoder
    """
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, AudioDecoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_vocoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "Vocoder":
    """Load the vocoder (for audio waveform generation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded Vocoder
    """
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import VOCODER_COMFY_KEYS_FILTER, VocoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_text_encoder(
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
) -> "GemmaTextEncoder":
    """Load the Gemma text encoder.
    Args:
        gemma_model_path: Path to Gemma model directory
        device: Device to load model on
        dtype: Data type for model weights
        load_in_8bit: Whether to load the Gemma model in 8-bit precision using bitsandbytes.
            When True, the model is loaded with device_map="auto" and the device argument
            is ignored for the Gemma backbone.
    Returns:
        Loaded GemmaTextEncoder
    """
    if not Path(gemma_model_path).is_dir():
        raise ValueError(f"Gemma model path is not a directory: {gemma_model_path}")

    # Use 8-bit loading path if requested
    if load_in_8bit:
        from ltx_trainer.gemma_8bit import load_8bit_gemma

        return load_8bit_gemma(gemma_model_path, dtype)

    # Standard loading path
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        GEMMA_LLM_KEY_OPS,
        GEMMA_MODEL_OPS,
        GemmaTextEncoderConfigurator,
        module_ops_from_gemma_root,
    )
    from ltx_core.text_encoders.gemma.config import resolve_gemma_checkpoint_config
    from ltx_core.utils import find_matching_file

    torch_device = _to_torch_device(device)

    gemma_model_folder = find_matching_file(str(gemma_model_path), "model*.safetensors").parent
    gemma_weight_paths = [str(p) for p in gemma_model_folder.rglob("*.safetensors")]
    weight_paths_t = tuple(gemma_weight_paths)
    gemma_cfg = resolve_gemma_checkpoint_config(weight_paths_t)

    text_encoder = SingleGPUModelBuilder(
        model_path=weight_paths_t,
        model_class_configurator=GemmaTextEncoderConfigurator,
        model_sd_ops=GEMMA_LLM_KEY_OPS,
        module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_root(str(gemma_model_path), gemma_cfg)),
    ).with_checkpoint_config(gemma_cfg).build(device=torch_device, dtype=dtype)

    return text_encoder


def load_embeddings_processor(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    gemma_model_path: str | Path | None = None,
    *,
    gemma_encode_stack_dims: dict[str, int] | None = None,
    require_matched_gemma_text_flat_dim: bool = False,
    flat_dim_bridge_rank: int | None = None,
) -> "EmbeddingsProcessor":
    """Load the embeddings processor (feature extractor + video/audio connectors).
    Args:
        checkpoint_path: Path to the LTX-2 safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
        gemma_model_path: Optional Gemma 4 weight tree (same layout as :func:`load_text_encoder`). When set,
            resolved Hugging Face Gemma ``config`` is merged into the LTX checkpoint metadata so the feature
            extractor ``flat_dim`` matches the text encoder width and depth (required for checkpoints trained with
            Gemma 4 26B and similar).

            When the checkpoint's ``video_aggregate_embed`` width (read from safetensors) differs from the Gemma
            stack's ``hidden_size * num_stack``, the merged config also sets ``ltx_checkpoint_text_flat_dim`` and
            ``ltx_experimental_flat_dim_bridge`` so weights load (random bridge Linear; prefer a Gemma tree that
            matches the checkpoint for best prompt fidelity).

        gemma_encode_stack_dims: Optional ``{"hidden_size", "num_stack"}`` merged into the resolved Gemma config
            (same meaning as ``ltx_encode_stack_dims`` in :mod:`ltx_core`). Use only when HF metadata disagrees with
            the real ``encode()`` stack.

        require_matched_gemma_text_flat_dim: When true, raise if Gemma flat width and checkpoint ``in_features``
            disagree instead of enabling the experimental bridge.
        flat_dim_bridge_rank: When set and the bridge is used, build a low-rank bottleneck instead of a dense
            ``Linear(flat_in, flat_ck)`` (see ``ltx_experimental_flat_dim_bridge_rank`` in ltx-core).
    Returns:
        Loaded EmbeddingsProcessor with feature extractor and connectors
    """
    from ltx_core.loader.helpers import peek_video_aggregate_embed_in_features, read_model_config
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        EmbeddingsProcessorConfigurator,
    )
    from ltx_core.text_encoders.gemma.config import resolve_gemma_checkpoint_config
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import _gemma_text_stack_dims
    from ltx_core.utils import find_matching_file

    torch_device = _to_torch_device(device)
    loader = SafetensorsModelStateDictLoader()

    builder = SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=EmbeddingsProcessorConfigurator,
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
    )
    if gemma_model_path is not None:
        gemma_folder = find_matching_file(str(gemma_model_path), "model*.safetensors").parent
        weight_paths_t = tuple(str(p) for p in gemma_folder.rglob("*.safetensors"))
        gemma_cfg = resolve_gemma_checkpoint_config(weight_paths_t)
        if gemma_encode_stack_dims is not None:
            gemma_cfg = {**gemma_cfg, "ltx_encode_stack_dims": dict(gemma_encode_stack_dims)}
        ltx_ck = read_model_config(str(checkpoint_path), loader)
        merged: dict = {**ltx_ck, "gemma_hf_config": gemma_cfg}
        flat_ck = peek_video_aggregate_embed_in_features(str(checkpoint_path))
        hs, ns = _gemma_text_stack_dims(gemma_cfg)
        gemma_flat = int(hs) * int(ns)
        if flat_ck is not None and flat_ck != gemma_flat:
            if require_matched_gemma_text_flat_dim:
                raise ValueError(
                    f"Gemma text flat_dim is {gemma_flat} (hidden_size×num_stack from merged HF config) but this LTX "
                    f"checkpoint expects video_aggregate_embed.in_features={flat_ck}. Use a Gemma release whose "
                    "stacked encode width matches this checkpoint, use an LTX checkpoint exported for your Gemma "
                    "tree, or set model.require_matched_gemma_text_flat_dim=false to allow the experimental "
                    "flat_dim bridge (then re-run process_captions with the same encoder + checkpoint so "
                    "conditions/*.pt align)."
                )
            merged["ltx_checkpoint_text_flat_dim"] = flat_ck
            merged["ltx_experimental_flat_dim_bridge"] = True
            allow_dense = os.environ.get("LTX_ALLOW_DENSE_FLAT_DIM_BRIDGE", "").lower() in ("1", "true", "yes")
            if flat_dim_bridge_rank is None and not allow_dense:
                default_rank = int(os.environ.get("LTX_DEFAULT_FLAT_DIM_BRIDGE_RANK", "32"))
                flat_dim_bridge_rank = default_rank
                logger.warning(
                    "Gemma/LTX flat_dim mismatch and no flat_dim_bridge_rank set — defaulting to rank %s. "
                    "Pass --flat-dim-bridge-rank explicitly or set LTX_ALLOW_DENSE_FLAT_DIM_BRIDGE=1 for the legacy "
                    "dense random bridge (not recommended).",
                    default_rank,
                )
            if flat_dim_bridge_rank is not None:
                merged["ltx_experimental_flat_dim_bridge_rank"] = int(flat_dim_bridge_rank)
            bridge_kind = (
                f"low-rank bottleneck (rank {int(flat_dim_bridge_rank)})"
                if flat_dim_bridge_rank is not None
                else "dense random Linear (very large in bf16)"
            )
            logger.warning(
                "Gemma encode flat_dim (%s) differs from LTX checkpoint video_aggregate_embed in_features (%s). "
                "Using experimental flat_dim bridge: %s. Re-run process_captions with the same encoder + checkpoint "
                "after changes so conditions/*.pt align; training applies connectors on top of cached tensors.",
                gemma_flat,
                flat_ck,
                bridge_kind,
            )
        builder = builder.with_checkpoint_config(merged)

    return builder.build(device=torch_device, dtype=dtype)


# =============================================================================
# Combined Component Loader
# =============================================================================


@dataclass
class LtxModelComponents:
    """Container for all LTX-2 model components."""

    transformer: "LTXModel"
    video_vae_encoder: "VideoEncoder | None" = None
    video_vae_decoder: "VideoDecoder | None" = None
    audio_vae_decoder: "AudioDecoder | None" = None
    vocoder: "Vocoder | None" = None
    text_encoder: "GemmaTextEncoder | None" = None
    scheduler: "LTX2Scheduler | None" = None


def load_model(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    with_video_vae_encoder: bool = False,
    with_video_vae_decoder: bool = True,
    with_audio_vae_decoder: bool = True,
    with_vocoder: bool = True,
    with_text_encoder: bool = True,
) -> LtxModelComponents:
    """
    Load LTX-2 model components from a safetensors checkpoint.
    This is a convenience function that loads multiple components at once.
    For loading individual components, use the dedicated functions:
    - load_transformer()
    - load_video_vae_encoder()
    - load_video_vae_decoder()
    - load_audio_vae_decoder()
    - load_vocoder()
    - load_text_encoder()
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        text_encoder_path: Path to Gemma model directory (required if with_text_encoder=True)
        device: Device to load models on ("cuda", "cpu", etc.)
        dtype: Data type for model weights
        with_video_vae_encoder: Whether to load the video VAE encoder (for preprocessing)
        with_video_vae_decoder: Whether to load the video VAE decoder (for inference/validation)
        with_audio_vae_decoder: Whether to load the audio VAE decoder
        with_vocoder: Whether to load the vocoder
        with_text_encoder: Whether to load the text encoder
    Returns:
        LtxModelComponents containing all loaded model components
    """
    from ltx_core.components.schedulers import LTX2Scheduler

    checkpoint_path = Path(checkpoint_path)

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading LTX-2 model from {checkpoint_path}")

    torch_device = _to_torch_device(device)

    # Load transformer
    logger.debug("Loading transformer...")
    transformer = load_transformer(checkpoint_path, torch_device, dtype)

    # Load video VAE encoder
    video_vae_encoder = None
    if with_video_vae_encoder:
        logger.debug("Loading video VAE encoder...")
        video_vae_encoder = load_video_vae_encoder(checkpoint_path, torch_device, dtype)

    # Load video VAE decoder
    video_vae_decoder = None
    if with_video_vae_decoder:
        logger.debug("Loading video VAE decoder...")
        video_vae_decoder = load_video_vae_decoder(checkpoint_path, torch_device, dtype)

    # Load audio VAE decoder
    audio_vae_decoder = None
    if with_audio_vae_decoder:
        logger.debug("Loading audio VAE decoder...")
        audio_vae_decoder = load_audio_vae_decoder(checkpoint_path, torch_device, dtype)

    # Load vocoder
    vocoder = None
    if with_vocoder:
        logger.debug("Loading vocoder...")
        vocoder = load_vocoder(checkpoint_path, torch_device, dtype)

    # Load text encoder
    text_encoder = None
    if with_text_encoder:
        if text_encoder_path is None:
            raise ValueError("text_encoder_path must be provided when with_text_encoder=True")
        logger.debug("Loading Gemma text encoder...")
        text_encoder = load_text_encoder(text_encoder_path, torch_device, dtype)

    # Create scheduler (stateless, no loading needed)
    scheduler = LTX2Scheduler()

    return LtxModelComponents(
        transformer=transformer,
        video_vae_encoder=video_vae_encoder,
        video_vae_decoder=video_vae_decoder,
        audio_vae_decoder=audio_vae_decoder,
        vocoder=vocoder,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
