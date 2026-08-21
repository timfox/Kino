import gc
import logging
import os

import torch

from ltx_core.components.noisers import Noiser
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
)
from ltx_core.model.audio_vae import AudioProcessor, encode_audio_tensor_for_inference
from ltx_core.model.transformer import Modality
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_core.text_encoders.gemma import GemmaTextEncoder
from ltx_core.tools import LatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import (
    decode_audio_from_file,
    decode_image,
    decode_video_from_file,
    get_videostream_fps,
    load_image_and_preprocess,
    resize_aspect_ratio_preserving,
    video_preprocess,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def parse_torch_cuda_index(spec: str | None) -> int | None:
    """Parse ``GOPEX_LTX_DEVICE`` / ``CUDA_DEVICE`` style string → CUDA index, or ``None`` if unset."""
    raw = (spec or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw.lower().startswith("cuda:"):
        tail = raw.split(":", 1)[1].strip()
        return int(tail) if tail.isdigit() else None
    return None


def resolve_ltx_compute_device(explicit: torch.device | None = None) -> torch.device:
    """Torch device for diffusion, upsampler, image/audio blocks, and the embeddings processor.

    Gemma may run elsewhere via :func:`resolve_prompt_gemma_device`. Video VAE decode may differ via
    :func:`resolve_video_decode_device`.
    """
    if explicit is not None:
        return explicit
    idx = parse_torch_cuda_index(os.environ.get("GOPEX_LTX_DEVICE") or os.environ.get("CUDA_DEVICE"))
    if idx is not None and torch.cuda.is_available():
        if idx < 0 or idx >= torch.cuda.device_count():
            logging.warning("GOPEX_LTX_DEVICE index %s out of range; using current_device.", idx)
            return get_device()
        return torch.device("cuda", idx)
    return get_device()


def resolve_video_decode_device(main_device: torch.device, raw: str | None) -> torch.device:
    """Where the **video VAE decoder** runs: same GPU as *main_device*, another CUDA device, or CPU.

    *raw* is typically from ``GOPEX_LTX_VIDEO_DECODE_DEVICE`` or a project field: ``\"\"`` (same as main),
    ``\"cpu\"``, ``\"cuda:1\"``, or ``\"1\"``.
    """
    s = (raw or "").strip()
    if not s:
        return main_device
    low = s.lower()
    if low in ("cpu", "host", "c"):
        return torch.device("cpu")
    if low.startswith("cuda:"):
        return torch.device(s)
    if s.isdigit() and torch.cuda.is_available():
        i = int(s)
        if 0 <= i < torch.cuda.device_count():
            return torch.device("cuda", i)
        logging.warning("Video decode CUDA index %s invalid; using main device %s.", s, main_device)
        return main_device
    if low in ("same", "main", "ltx", "auto"):
        return main_device
    logging.warning("Unrecognized video decode device %r; using main device %s.", s, main_device)
    return main_device


def resolve_prompt_gemma_device(main_device: torch.device, spec: str | None) -> torch.device:
    """CUDA/CPU device for **Gemma** inside :class:`~ltx_pipelines.utils.blocks.PromptEncoder`.

    When *spec* is empty, ``GOPEX_LTX_GEMMA_DEVICE`` is consulted; if still empty, returns *main_device*
    (legacy single-GPU: Gemma and LTX share one card).

    Accepts the same strings as :func:`resolve_video_decode_device` (``cpu``, ``cuda:0``, ``1``, …).
    After Gemma runs, hidden states are moved to *main_device* for the checkpoint embeddings processor.
    """
    raw = (spec or "").strip() or (os.environ.get("GOPEX_LTX_GEMMA_DEVICE") or "").strip()
    return resolve_video_decode_device(main_device, raw)


def run_video_decode(
    video_decoder,
    latent: torch.Tensor,
    tiling_config,
    generator: torch.Generator,
    *,
    seed: int,
    dtype: torch.dtype,
    main_device: torch.device,
    decode_device: torch.device,
):
    """Decode *latent* with *video_decoder* (built for *decode_device*), moving tensors when devices differ."""
    if decode_device == main_device:
        return video_decoder(latent, tiling_config, generator)
    latent_d = latent.detach().to(device=decode_device, dtype=dtype)
    gen_d = torch.Generator(device=decode_device).manual_seed(seed)
    return video_decoder(latent_d, tiling_config, gen_d)


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    try:
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        logging.warning("Host empty cache cleanup failed; ignoring.", exc_info=True)


def _conform_latent_length(latent: torch.Tensor, expected_frames_count: int) -> torch.Tensor:
    actual_frames = latent.shape[2]
    if actual_frames > expected_frames_count:
        latent = latent[:, :, :expected_frames_count]
    elif actual_frames < expected_frames_count:
        shape_as_list = list(latent.shape)
        shape_as_list[2] = expected_frames_count - actual_frames
        pad = torch.zeros(
            shape_as_list,
            device=latent.device,
            dtype=latent.dtype,
        )
        latent = torch.cat([latent, pad], dim=2)
    return latent


def video_latent_from_file(
    video_encoder: VideoEncoder,
    file_path: str,
    output_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    start_time: float = 0.0,
    max_duration: float | None = None,
    tiling_config: TilingConfig | None = None,
) -> torch.Tensor | None:
    """Load video from a file, and construct the video latent conforming to video output shape.
    Args:
        video_encoder: Model used to encode pixel frames to latent space.
        file_path: Path to the video file.
        output_shape: Target pixel shape (height, width, frames, fps) for the conditioning.
        device: Device to run the encoder and hold tensors on.
        dtype: Dtype for the output latents.
        start_time: Start time in seconds to begin reading the video (default 0.0).
        max_duration: Maximum duration in seconds. If None, uses output_shape.frames at
            output_shape.fps (default None).
        tiling_config: Tiling configuration for the encoder. Defaults to TilingConfig.default().
    Returns:
        Encoded video latents of shape (1, C, T, H, W) with T = required_latent_frames, or
        None (currently this function always returns a tensor).
    """
    fps = get_videostream_fps(file_path)
    if fps != output_shape.fps:
        raise ValueError(f"Input video FPS {fps} does not match output FPS {output_shape.fps}, not supported")
    max_duration = max_duration or output_shape.frames / fps
    frame_gen = decode_video_from_file(path=file_path, device=device, start_time=start_time, max_duration=max_duration)
    frames = video_preprocess(frame_gen, output_shape.height, output_shape.width, dtype, device)
    latents = video_encoder.tiled_encode(frames, tiling_config or TilingConfig.default())
    required_latent_frames = VideoLatentShape.from_pixel_shape(output_shape).frames
    return _conform_latent_length(latents, required_latent_frames)


def audio_latent_from_file(
    audio_encoder: torch.nn.Module,
    file_path: str,
    output_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> torch.Tensor | None:
    """Load audio from a file, and construct the audio latent conforming to video output shape.
    Args:
        audio_encoder: Model used to encode audio to latent space.
        file_path: Path to the audio or video file containing an audio stream.
        output_shape: Target video pixel shape; used to derive required latent frames
            and, when max_duration is None, the audio duration (output_shape.frames / fps).
        device: Device to run the encoder and hold tensors on.
        dtype: Dtype for the output latents.
        start_time: Start time in seconds to begin reading the audio (default 0.0).
        max_duration: Maximum duration in seconds. If None, uses the full span implied
            by output_shape (default None).
    Returns:
        Encoded audio latents of shape (1, C, T, ...) with T = required_latent_frames, or
        None if the file has no audio stream.
    """
    max_duration = max_duration or output_shape.frames / output_shape.fps
    audio_in = decode_audio_from_file(file_path, device, start_time, max_duration)
    if audio_in is None:
        return None
    try:
        processor = AudioProcessor(
            target_sample_rate=audio_encoder.sample_rate,
            mel_bins=audio_encoder.mel_bins,
            mel_hop_length=audio_encoder.mel_hop_length,
            n_fft=audio_encoder.n_fft,
        ).to(device=device)
    except Exception:
        processor = None
    return encode_audio_tensor_for_inference(
        audio_in, audio_encoder, processor, output_shape
    ).to(device, dtype)


def combined_image_conditionings(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    """Create a list of conditionings by replacing the latent at the first frame with the encoded image if present
    and using other encoded images as the keyframe conditionings."""
    conditionings = []
    for img in images:
        image = load_image_and_preprocess(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
        )
        encoded_image = video_encoder(image)
        if img.frame_idx == 0:
            conditioning = VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=img.strength,
                latent_idx=0,
            )
        else:
            conditioning = VideoConditionByKeyframeIndex(
                keyframes=encoded_image,
                strength=img.strength,
                frame_idx=img.frame_idx,
            )
        conditionings.append(conditioning)
    return conditionings


def image_conditionings_by_replacing_latent(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    conditionings = []
    for img in images:
        image = load_image_and_preprocess(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=img.strength,
                latent_idx=img.frame_idx,
            )
        )

    return conditionings


def image_conditionings_by_adding_guiding_latent(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    conditionings = []
    for img in images:
        image = load_image_and_preprocess(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByKeyframeIndex(keyframes=encoded_image, frame_idx=img.frame_idx, strength=img.strength)
        )
    return conditionings


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> LatentState:
    """Create a noised latent state from empty state, conditionings, and noiser.
    Creates an empty latent state, applies conditionings, and then adds noise
    using the provided noiser. Returns the final noised state ready for diffusion.
    """
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)
    state = noiser(state, noise_scale)

    return state


def state_with_conditionings(
    latent_state: LatentState, conditioning_items: list[ConditioningItem], latent_tools: LatentTools
) -> LatentState:
    """Apply a list of conditionings to a latent state.
    Iterates through the conditioning items and applies each one to the latent
    state in sequence. Returns the modified state with all conditionings applied.
    """
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)

    return latent_state


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Blend denoised output with clean state based on mask."""
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(
    state: LatentState,
    context: torch.Tensor,
    sigma: torch.Tensor,
    enabled: bool = True,
) -> Modality:
    """Create a Modality from a latent state.
    Constructs a Modality object with the latent state's data, timesteps derived
    from the denoise mask and sigma, positions, and the provided context.
    """
    return Modality(
        enabled=enabled,
        # Samplers may keep their arithmetic in fp32, while native LTX 2.x
        # transformer weights are bf16. Normalize at the model boundary so
        # resident and streaming/offload modes behave identically.
        latent=state.latent.to(device=context.device, dtype=torch.bfloat16),
        sigma=sigma,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context.to(device=context.device, dtype=torch.bfloat16),
        context_mask=None,
        attention_mask=state.attention_mask,
    )


def timesteps_from_mask(denoise_mask: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Compute timesteps from a denoise mask and sigma value.
    Multiplies the denoise mask by sigma to produce timesteps for each position
    in the latent state. Areas where the mask is 0 will have zero timesteps.
    When sigma is ``(B,)`` it is reshaped to ``(B, 1, ...)`` so the batch
    dimension aligns correctly with ``denoise_mask``.
    """
    if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
        sigma = sigma.view(-1, *([1] * (denoise_mask.dim() - 1)))
    return denoise_mask * sigma


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    """Clean a response from curly quotes and leading non-letter characters which Gemma tends to insert."""
    text = text.translate(_UNICODE_REPLACEMENTS)

    # Remove leading non-letter characters
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def build_reference_aware_prompt(prompt: str, reference_hint: str | None = None) -> str:
    """Augment *prompt* with optional free-text reference context (Gemma preprocessing)."""
    if not reference_hint:
        return prompt
    hint = reference_hint.strip()
    if not hint:
        return prompt
    return f"{prompt.rstrip()}\n\n[Reference]\n{hint}"


def generate_enhanced_prompt(
    text_encoder: GemmaTextEncoder,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,
    seed: int = 42,
) -> str:
    """Generate an enhanced prompt from a text encoder and a prompt."""
    image = None
    if image_path:
        image = decode_image(image_path=image_path)
        image = torch.tensor(image)
        image = resize_aspect_ratio_preserving(image, image_long_side).to(torch.uint8)
        prompt = text_encoder.enhance_i2v(prompt, image, seed=seed)
    else:
        prompt = text_encoder.enhance_t2v(prompt, seed=seed)
    logging.info(f"Enhanced prompt: {prompt}")
    return clean_response(prompt)


def assert_resolution(height: int, width: int, is_two_stage: bool) -> None:
    """Assert that the resolution is divisible by the required divisor.
    For two-stage pipelines, the resolution must be divisible by 64.
    For one-stage pipelines, the resolution must be divisible by 32.
    """
    divisor = 64 if is_two_stage else 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {'two-stage' if is_two_stage else 'one-stage'} pipelines, "
            f"height and width must be multiples of {divisor}."
        )
