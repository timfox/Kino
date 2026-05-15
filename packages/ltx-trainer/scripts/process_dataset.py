#!/usr/bin/env python3

"""
Preprocess a video dataset by computing video clips latents and text captions embeddings.
This script provides a command-line interface for preprocessing video datasets by computing
latent representations of video clips and text embeddings of their captions. The preprocessed
data can be used to accelerate training of video generation models and to save GPU memory.
Basic usage:
    python scripts/process_dataset.py /path/to/dataset.json --resolution-buckets 768x768x49 \\
        --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
        --flat-dim-bridge-rank 512
The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.
"""

from pathlib import Path

import typer
from decode_latents import LatentsDecoder
from process_captions import compute_captions_embeddings
from process_videos import (
    _ALLOWED_HDR_TRANSFER,
    _ALLOWED_HDR_VAE_ENCODING,
    _parse_latent_save_dtype,
    compute_latents,
    compute_scaled_resolution_buckets,
    parse_resolution_buckets,
)
from rich.console import Console

from ltx_trainer import logger
from ltx_trainer.gpu_utils import free_gpu_memory_context

console = Console()

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Preprocess a video dataset by computing video clips latents and text captions embeddings. "
    "The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.",
)


def preprocess_dataset(  # noqa: PLR0913
    dataset_file: str,
    caption_column: str,
    video_column: str,
    resolution_buckets: list[tuple[int, int, int]],
    batch_size: int,
    output_dir: str | None,
    lora_trigger: str | None,
    vae_tiling: bool,
    decode: bool,
    model_path: str,
    text_encoder_path: str,
    device: str,
    remove_llm_prefixes: bool = False,
    reference_column: str | None = None,
    reference_downscale_factor: int = 1,
    with_audio: bool = False,
    load_text_encoder_in_8bit: bool = False,
    flat_dim_bridge_rank: int | None = None,
    hdr_ingest: bool = False,
    hdr_transfer: str = "auto",
    hdr_synth_bracket_ev: str | None = None,
    hdr_vae_encoding: str = "reinhard",
    latent_save_dtype: str = "float32",
) -> None:
    """Run the preprocessing pipeline with the given arguments."""
    # Validate dataset file
    _validate_dataset_file(dataset_file)

    ht = hdr_transfer.lower().strip()
    if ht not in _ALLOWED_HDR_TRANSFER:
        raise ValueError(
            f"Unknown hdr_transfer {hdr_transfer!r}; expected one of: {', '.join(sorted(_ALLOWED_HDR_TRANSFER))}"
        )
    try:
        latent_dtype = _parse_latent_save_dtype(latent_save_dtype)
    except typer.BadParameter as exc:
        raise ValueError(str(exc)) from exc

    if hdr_synth_bracket_ev and not hdr_ingest:
        raise ValueError("hdr_synth_bracket_ev requires hdr_ingest=True")

    hve = hdr_vae_encoding.lower().strip()
    if hve not in _ALLOWED_HDR_VAE_ENCODING:
        raise ValueError(
            f"Unknown hdr_vae_encoding {hdr_vae_encoding!r}; expected one of: "
            f"{', '.join(sorted(_ALLOWED_HDR_VAE_ENCODING))}"
        )
    if hve != "reinhard" and not hdr_ingest:
        raise ValueError(
            "hdr_vae_encoding other than 'reinhard' requires hdr_ingest=True (pu21, logc3, or lf_log1p with HDR ingest)"
        )

    # Set up output directories
    output_base = Path(output_dir) if output_dir else Path(dataset_file).parent / ".precomputed"
    conditions_dir = output_base / "conditions"
    latents_dir = output_base / "latents"

    if lora_trigger:
        logger.info(f'LoRA trigger word "{lora_trigger}" will be prepended to all captions')

    with free_gpu_memory_context():
        # Process captions using the dedicated function
        compute_captions_embeddings(
            dataset_file=dataset_file,
            output_dir=str(conditions_dir),
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            caption_column=caption_column,
            media_column=video_column,
            lora_trigger=lora_trigger,
            remove_llm_prefixes=remove_llm_prefixes,
            batch_size=batch_size,
            device=device,
            load_in_8bit=load_text_encoder_in_8bit,
            flat_dim_bridge_rank=flat_dim_bridge_rank,
        )

    # Process videos using the dedicated function
    audio_latents_dir = None
    if with_audio:
        logger.info("Audio preprocessing enabled - will extract and encode audio from videos")
        audio_latents_dir = output_base / "audio_latents"

    with free_gpu_memory_context():
        compute_latents(
            dataset_file=dataset_file,
            video_column=video_column,
            resolution_buckets=resolution_buckets,
            output_dir=str(latents_dir),
            model_path=model_path,
            batch_size=batch_size,
            device=device,
            vae_tiling=vae_tiling,
            with_audio=with_audio,
            audio_output_dir=str(audio_latents_dir) if audio_latents_dir else None,
            hdr_ingest=hdr_ingest,
            hdr_transfer=ht,
            hdr_synth_bracket_ev=hdr_synth_bracket_ev,
            hdr_vae_encoding=hve,
            latent_save_dtype=latent_dtype,
        )

        # Process reference videos if reference_column is provided
        if reference_column:
            # Validate: scaled references with multiple buckets can cause ambiguous bucket matching
            if reference_downscale_factor > 1 and len(resolution_buckets) > 1:
                raise ValueError(
                    "When using --reference-downscale-factor > 1, only a single resolution bucket is supported. "
                    "Using multiple buckets with scaled references can cause ambiguous bucket matching "
                    "(e.g., a 512x256 reference could match either the scaled-down 1024x512 bucket or the 512x256 "
                    "bucket). Please use a single resolution bucket or set --reference-downscale-factor to 1."
                )

            # Calculate and validate scaled resolution buckets for reference videos
            reference_buckets = compute_scaled_resolution_buckets(resolution_buckets, reference_downscale_factor)

            if reference_downscale_factor > 1:
                logger.info(
                    f"Processing reference videos for IC-LoRA training at 1/{reference_downscale_factor} resolution..."
                )
                logger.info(f"Reference resolution buckets: {reference_buckets}")
            else:
                logger.info("Processing reference videos for IC-LoRA training...")

            reference_latents_dir = output_base / "reference_latents"

            compute_latents(
                dataset_file=dataset_file,
                main_media_column=video_column,
                video_column=reference_column,
                resolution_buckets=reference_buckets,
                output_dir=str(reference_latents_dir),
                model_path=model_path,
                batch_size=batch_size,
                device=device,
                vae_tiling=vae_tiling,
                hdr_ingest=hdr_ingest,
                hdr_transfer=ht,
                hdr_synth_bracket_ev=hdr_synth_bracket_ev,
                hdr_vae_encoding=hve,
                latent_save_dtype=latent_dtype,
            )

    # Handle decoding if requested (for verification)
    if decode:
        logger.info("Decoding latents for verification...")

        decoder = LatentsDecoder(
            model_path=model_path,
            device=device,
            vae_tiling=vae_tiling,
            with_audio=with_audio,
        )
        decoder.decode(latents_dir, output_base / "decoded_videos")

        # Also decode reference videos if they exist
        if reference_column:
            reference_latents_dir = output_base / "reference_latents"
            if reference_latents_dir.exists():
                logger.info("Decoding reference videos...")
                decoder.decode(reference_latents_dir, output_base / "decoded_reference_videos")

        # Decode audio latents if they exist
        if with_audio and audio_latents_dir and audio_latents_dir.exists():
            logger.info("Decoding audio latents...")
            decoder.decode_audio(audio_latents_dir, output_base / "decoded_audio")

    # Print summary
    logger.info(f"Dataset preprocessing complete! Results saved to {output_base}")
    if reference_column:
        logger.info("Reference videos processed and saved to reference_latents/ directory for IC-LoRA training")
    if with_audio:
        logger.info("Audio latents saved to audio_latents/ directory for audio-video training")


def _validate_dataset_file(dataset_path: str) -> None:
    """Validate that the dataset file exists and has the correct format."""
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")

    if not dataset_file.is_file():
        raise ValueError(f"Dataset path must be a file, not a directory: {dataset_file}")

    if dataset_file.suffix.lower() not in [".csv", ".json", ".jsonl"]:
        raise ValueError(f"Dataset file must be CSV, JSON, or JSONL format: {dataset_file}")


@app.command()
def main(  # noqa: PLR0913
    dataset_path: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing captions and video paths",
    ),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF;WxHxF;..." (e.g. "768x768x25;512x512x49")',
    ),
    model_path: str = typer.Option(
        ...,
        help="Path to LTX-2 checkpoint (.safetensors file)",
    ),
    text_encoder_path: str = typer.Option(
        ...,
        help="Path to Gemma text encoder directory",
    ),
    caption_column: str = typer.Option(
        default="caption",
        help="Column name containing captions in the dataset JSON/JSONL/CSV file",
    ),
    video_column: str = typer.Option(
        default="media_path",
        help="Column name containing video paths in the dataset JSON/JSONL/CSV file",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size for preprocessing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    output_dir: str | None = typer.Option(
        default=None,
        help="Output directory (defaults to .precomputed in dataset directory)",
    ),
    lora_trigger: str | None = typer.Option(
        default=None,
        help="Optional trigger word to prepend to each caption (activates the LoRA during inference)",
    ),
    decode: bool = typer.Option(
        default=False,
        help="Decode and save latents after encoding (videos and audio) for verification",
    ),
    remove_llm_prefixes: bool = typer.Option(
        default=False,
        help="Remove LLM prefixes from captions",
    ),
    reference_column: str | None = typer.Option(
        default=None,
        help="Column name containing reference video paths (for video-to-video training)",
    ),
    with_audio: bool = typer.Option(
        default=False,
        help="Extract and encode audio from video files",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the Gemma text encoder in 8-bit precision to save GPU memory (requires bitsandbytes)",
    ),
    flat_dim_bridge_rank: int | None = typer.Option(
        default=None,
        help="When Gemma stacked width mismatches the LTX checkpoint, use low-rank experimental bridge with this rank "
        "(must match training YAML model.flat_dim_bridge_rank; see tools/LTX_TRAINER.md §6 in GopexLLC).",
    ),
    reference_downscale_factor: int = typer.Option(
        default=1,
        help="Downscale factor for reference video resolution. When > 1, reference videos are processed at "
        "1/n resolution (e.g., 2 means half resolution). Used for efficient IC-LoRA training.",
    ),
    hdr_ingest: bool = typer.Option(
        default=False,
        help="HDR decode to scene-linear float32; save hdr_latent per clip; tone-map for VAE input",
    ),
    hdr_transfer: str = typer.Option(
        default="auto",
        help="Color transfer for HDR linearization: auto, pq, hlg, srgb, or linear",
    ),
    latent_save_dtype: str = typer.Option(
        default="float32",
        help="Torch dtype for saved VAE latents on disk: float32, bfloat16, or float16",
    ),
    hdr_synth_bracket_ev: str | None = typer.Option(
        default=None,
        help='LatentHDR-style synthetic γ-LDR stack spec "ev_min:ev_max:step" (e.g. "-7:5:1"); requires --hdr-ingest',
    ),
    hdr_vae_encoding: str = typer.Option(
        default="reinhard",
        help="With --hdr-ingest: VAE pixel encoding: reinhard | pu21 | logc3 | lf_log1p (LF-Diff log tonemap μ=5000)",
    ),
) -> None:
    """Preprocess a video dataset by computing and saving latents and text embeddings.
    The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.
    This script is designed for LTX-2 models which use the Gemma text encoder.
    Examples:
        # Process a dataset with LTX-2 model
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma
        # Process dataset with custom column names
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --caption-column "text" --video-column "video_path"
        # Process dataset with reference videos for IC-LoRA training
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --reference-column "reference_path"
        # Process dataset with scaled reference videos (half resolution) for efficient IC-LoRA
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --reference-column "reference_path" --reference-downscale-factor 2
        # Process dataset with audio for audio-video training
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x512x97 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --with-audio
    """
    parsed_resolution_buckets = parse_resolution_buckets(resolution_buckets)

    if len(parsed_resolution_buckets) > 1:
        logger.warning(
            "Using multiple resolution buckets. "
            "When training with multiple resolution buckets, you must use a batch size of 1."
        )

    # Validate reference_downscale_factor
    if reference_downscale_factor < 1:
        raise typer.BadParameter("--reference-downscale-factor must be >= 1")

    if reference_downscale_factor > 1 and not reference_column:
        logger.warning("--reference-downscale-factor specified but no --reference-column provided. Ignoring.")

    if hdr_synth_bracket_ev and not hdr_ingest:
        raise typer.BadParameter("--hdr-synth-bracket-ev requires --hdr-ingest")

    hve = hdr_vae_encoding.lower().strip()
    if hve not in _ALLOWED_HDR_VAE_ENCODING:
        raise typer.BadParameter(
            f"Unknown --hdr-vae-encoding {hdr_vae_encoding!r}; expected one of: "
            f"{', '.join(sorted(_ALLOWED_HDR_VAE_ENCODING))}"
        )
    if hve != "reinhard" and not hdr_ingest:
        raise typer.BadParameter("--hdr-vae-encoding pu21, logc3, or lf_log1p requires --hdr-ingest")

    preprocess_dataset(
        dataset_file=dataset_path,
        caption_column=caption_column,
        video_column=video_column,
        resolution_buckets=parsed_resolution_buckets,
        batch_size=batch_size,
        output_dir=output_dir,
        lora_trigger=lora_trigger,
        vae_tiling=vae_tiling,
        decode=decode,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        device=device,
        remove_llm_prefixes=remove_llm_prefixes,
        reference_column=reference_column,
        reference_downscale_factor=reference_downscale_factor,
        with_audio=with_audio,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        flat_dim_bridge_rank=flat_dim_bridge_rank,
        hdr_ingest=hdr_ingest,
        hdr_transfer=hdr_transfer,
        hdr_synth_bracket_ev=hdr_synth_bracket_ev,
        hdr_vae_encoding=hve,
        latent_save_dtype=latent_save_dtype,
    )


if __name__ == "__main__":
    app()
