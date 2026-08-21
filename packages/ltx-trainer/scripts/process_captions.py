#!/usr/bin/env python

"""
Compute text embeddings for video generation training.
This module provides functionality for processing text captions, including:
- Loading captions from various file formats (CSV, JSON, JSONL)
- Cleaning and preprocessing text (removing LLM prefixes, adding ID tokens)
- CaptionsDataset for caption-only preprocessing workflows
Can be used as a standalone script:
    python scripts/process_captions.py dataset.json --output-dir /path/to/output \
        --model-source /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer
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
from torch.utils.data import DataLoader, Dataset
from transformers.utils.logging import disable_progress_bar

from ltx_trainer import logger
from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder
from ltx_trainer.nvml_safe_cuda import apply_nvml_safe_cuda_patches, embeddings_processor_device

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

disable_progress_bar()

# Common phrases that LLMs often add to captions that we might want to remove
COMMON_BEGINNING_PHRASES: tuple[str, ...] = (
    "This video",
    "The video",
    "This clip",
    "The clip",
    "The animation",
    "This image",
    "The image",
    "This picture",
    "The picture",
)

COMMON_CONTINUATION_WORDS: tuple[str, ...] = (
    "shows",
    "depicts",
    "features",
    "captures",
    "highlights",
    "introduces",
    "presents",
)

COMMON_LLM_START_PHRASES: tuple[str, ...] = (
    "In the video,",
    "In this video,",
    "In this video clip,",
    "In the clip,",
    "Caption:",
    *(
        f"{beginning} {continuation}"
        for beginning in COMMON_BEGINNING_PHRASES
        for continuation in COMMON_CONTINUATION_WORDS
    ),
)

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process text captions and save embeddings for video generation training.",
)


def _shard_relpaths_from_dir(root: Path) -> set[str]:
    """Collect relative .pt paths under ``root`` using scandir (flat clips layout)."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        return set()
    clips = root / "ltx_manifest" / "clips"
    scan_root = clips if clips.is_dir() else root
    out: set[str] = set()
    try:
        with os.scandir(scan_root) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".pt"):
                    if scan_root is clips:
                        out.add(str(Path("ltx_manifest") / "clips" / entry.name))
                    else:
                        try:
                            out.add(str(Path(entry.path).relative_to(root)))
                        except ValueError:
                            out.add(entry.name)
    except OSError:
        return set()
    return out


class CaptionsDataset(Dataset):
    """
    Dataset for processing text captions only.
    This dataset is designed for caption preprocessing workflows where you only need
    to process text without loading videos. Useful for:
    - Precomputing text embeddings
    - Caption cleaning and preprocessing
    - Text-only preprocessing pipelines
    """

    def __init__(
        self,
        dataset_file: str | Path,
        caption_column: str,
        media_column: str = "media_path",
        lora_trigger: str | None = None,
        remove_llm_prefixes: bool = False,
    ) -> None:
        """
        Initialize the captions dataset.
        Args:
            dataset_file: Path to CSV/JSON/JSONL metadata file
            caption_column: Column name for captions in the metadata file
            media_column: Column name for media paths (used for output naming)
            lora_trigger: Optional trigger word to prepend to each caption
            remove_llm_prefixes: Whether to remove common LLM-generated prefixes
        """
        super().__init__()

        self.dataset_file = Path(dataset_file)
        self.caption_column = caption_column
        self.media_column = media_column
        self.lora_trigger = f"{lora_trigger.strip()} " if lora_trigger else ""

        # Load captions with their corresponding output embedding paths
        self.caption_data = self._load_caption_data()

        # Convert to lists for indexing
        self.output_paths = list(self.caption_data.keys())
        self.prompts = list(self.caption_data.values())

        # Clean LLM start phrases if requested
        if remove_llm_prefixes:
            self._clean_llm_prefixes()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single caption with optional trigger word prepended and output path."""
        prompt = self.lora_trigger + self.prompts[index]
        try:
            from ltx_trainer.forte.prompt_bridge import maybe_refine_ltx_prompt

            prompt = maybe_refine_ltx_prompt(prompt)
        except ImportError:
            pass
        return {
            "prompt": prompt,
            "output_path": self.output_paths[index],
            "index": index,
        }

    def filter_shards(
        self,
        *,
        conditions_dir: Path,
        latents_dir: Path | None = None,
        skip_existing: bool = False,
        require_latents: bool = False,
    ) -> tuple[int, int, int]:
        """Drop rows with existing conditions and/or missing latents. Returns kept, skipped_existing, skipped_no_latent."""
        cond_root = Path(conditions_dir).expanduser().resolve()
        latent_root = Path(latents_dir).expanduser().resolve() if require_latents and latents_dir else None
        if skip_existing or require_latents:
            logger.info(
                "Caption shard filter: scanning %s manifest rows (skip_existing=%s require_latents=%s)",
                f"{len(self.output_paths):,}",
                skip_existing,
                latent_root is not None,
            )

        kept_paths: list[str] = []
        kept_prompts: list[str] = []
        skipped_existing = 0
        skipped_no_latent = 0
        total = len(self.output_paths)
        for idx, (rel_path, prompt) in enumerate(zip(self.output_paths, self.prompts, strict=True)):
            if skip_existing and (cond_root / rel_path).is_file():
                skipped_existing += 1
                continue
            if latent_root is not None and not (latent_root / rel_path).is_file():
                skipped_no_latent += 1
                continue
            kept_paths.append(rel_path)
            kept_prompts.append(prompt)
            if (idx + 1) % 10000 == 0:
                logger.info(
                    "Caption shard filter: scanned %s / %s rows (%s kept, %s existing, %s no latent)",
                    f"{idx + 1:,}",
                    f"{total:,}",
                    f"{len(kept_paths):,}",
                    f"{skipped_existing:,}",
                    f"{skipped_no_latent:,}",
                )
        self.output_paths = kept_paths
        self.prompts = kept_prompts
        return len(kept_paths), skipped_existing, skipped_no_latent

    def _load_caption_data(self) -> dict[str, str]:
        """Load captions and compute their output embedding paths."""
        if self.dataset_file.suffix == ".csv":
            return self._load_caption_data_from_csv()
        elif self.dataset_file.suffix == ".json":
            return self._load_caption_data_from_json()
        elif self.dataset_file.suffix == ".jsonl":
            return self._load_caption_data_from_jsonl()
        else:
            raise ValueError("Expected `dataset_file` to be a path to a CSV, JSON, or JSONL file.")

    def _load_caption_data_from_csv(self) -> dict[str, str]:
        """Load captions from a CSV file and compute output embedding paths."""
        df = pd.read_csv(self.dataset_file)

        if self.caption_column not in df.columns:
            raise ValueError(f"Column '{self.caption_column}' not found in CSV file")
        if self.media_column not in df.columns:
            raise ValueError(f"Column '{self.media_column}' not found in CSV file")

        caption_data = {}
        for _, row in df.iterrows():
            media_path = Path(row[self.media_column].strip())
            # Convert media path to embedding output path (same structure, .pt extension)
            output_path = str(media_path.with_suffix(".pt"))
            caption_data[output_path] = row[self.caption_column]

        return caption_data

    def _load_caption_data_from_json(self) -> dict[str, str]:
        """Load captions from a JSON file and compute output embedding paths."""
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")

        caption_data = {}
        for entry in data:
            if self.caption_column not in entry:
                raise ValueError(f"Key '{self.caption_column}' not found in JSON entry: {entry}")
            if self.media_column not in entry:
                raise ValueError(f"Key '{self.media_column}' not found in JSON entry: {entry}")

            media_path = Path(entry[self.media_column].strip())
            # Convert media path to embedding output path (same structure, .pt extension)
            output_path = str(media_path.with_suffix(".pt"))
            caption_data[output_path] = entry[self.caption_column]

        return caption_data

    def _load_caption_data_from_jsonl(self) -> dict[str, str]:
        """Load captions from a JSONL file and compute output embedding paths."""
        caption_data = {}
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                if self.caption_column not in entry:
                    raise ValueError(f"Key '{self.caption_column}' not found in JSONL entry: {entry}")
                if self.media_column not in entry:
                    raise ValueError(f"Key '{self.media_column}' not found in JSONL entry: {entry}")

                media_path = Path(entry[self.media_column].strip())
                # Convert media path to embedding output path (same structure, .pt extension)
                output_path = str(media_path.with_suffix(".pt"))
                caption_data[output_path] = entry[self.caption_column]

        return caption_data

    def _clean_llm_prefixes(self) -> None:
        """Remove common LLM-generated prefixes from captions."""
        for i in range(len(self.prompts)):
            self.prompts[i] = self.prompts[i].strip()
            for phrase in COMMON_LLM_START_PHRASES:
                if self.prompts[i].startswith(phrase):
                    self.prompts[i] = self.prompts[i].removeprefix(phrase).strip()
                    break


def _resolve_embeddings_processor_device(
    *,
    device: str,
    embeddings_device: str | None,
    flat_dim_bridge_rank: int | None,
    load_in_8bit: bool,
) -> str:
    """Pick device for LTX EmbeddingsProcessor (feature_extractor), keeping Gemma on ``device`` when tight."""
    if embeddings_device is not None:
        return embeddings_processor_device(embeddings_device)
    force_cpu = os.environ.get("GOPEX_EMBEDDINGS_PROCESSOR_ON_CPU", "").lower() in ("1", "true", "yes")
    force_gpu = os.environ.get("GOPEX_EMBEDDINGS_PROCESSOR_ON_GPU", "").lower() in ("1", "true", "yes")
    if force_gpu:
        return embeddings_processor_device(device)
    if force_cpu:
        proc = "cpu"
    elif (
        embeddings_device is None
        and flat_dim_bridge_rank is None
        and os.environ.get("LTX_ALLOW_DENSE_FLAT_DIM_BRIDGE", "").lower() in ("1", "true", "yes")
    ):
        proc = "cpu"
        logger.warning(
            "Dense flat_dim bridge: embeddings processor on CPU (~30GB GPU avoided). "
            "Gemma stays on %s.",
            device,
        )
    elif str(device).startswith("cuda") and not load_in_8bit:
        # Gemma 4 31B bf16 alone uses ~62–65 GiB; native/folded LTX feature_extractor cannot fit on the same GPU.
        proc = "cpu"
        logger.warning(
            "Gemma 31B bf16 on %s — loading LTX embeddings processor on CPU (avoids OOM). "
            "Set GOPEX_EMBEDDINGS_PROCESSOR_ON_GPU=1 only if you use 8-bit Gemma or a second GPU.",
            device,
        )
    else:
        proc = embeddings_processor_device(device)
    return proc


def compute_captions_embeddings(  # noqa: PLR0913
    dataset_file: str | Path,
    output_dir: str,
    model_path: str,
    text_encoder_path: str,
    caption_column: str = "caption",
    media_column: str = "media_path",
    lora_trigger: str | None = None,
    remove_llm_prefixes: bool = False,
    batch_size: int = 8,
    device: str = "cuda",
    load_in_8bit: bool = False,
    flat_dim_bridge_rank: int | None = None,
    skip_existing: bool = False,
    require_latents: bool = False,
    latents_dir: str | Path | None = None,
    embeddings_device: str | None = None,
) -> None:
    """
    Process captions and save text embeddings.
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL) containing captions and media paths
        output_dir: Directory to save embeddings
        model_path: Path to LTX-2 checkpoint (.safetensors)
        text_encoder_path: Path to Gemma text encoder directory
        caption_column: Column name containing captions in the metadata file
        media_column: Column name containing media paths (used for output naming)
        lora_trigger: Optional trigger word to prepend to each caption
        remove_llm_prefixes: Whether to remove common LLM-generated prefixes
        batch_size: Batch size for processing
        device: Device to use for computation
        load_in_8bit: Whether to load the Gemma text encoder in 8-bit precision
        flat_dim_bridge_rank: When Gemma/LTX flat_dim mismatch enables the experimental bridge, use this bottleneck
            rank (same as training YAML ``model.flat_dim_bridge_rank``); omit for dense bridge.
        skip_existing: Skip captions whose output ``.pt`` already exists (resume interrupted runs).
        require_latents: Only embed captions that already have a matching latent shard under ``latents_dir``.
        latents_dir: Latents root (sibling of ``conditions/`` under precomputed); required when ``require_latents``.
        embeddings_device: Device for LTX embeddings processor (defaults to ``device``; use ``cpu`` for dense bridge).
    """

    console = Console()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    latents_root = Path(latents_dir).expanduser().resolve() if latents_dir else None
    if require_latents and latents_root is None:
        raise ValueError("require_latents=True requires latents_dir")

    # Create dataset
    dataset = CaptionsDataset(
        dataset_file=dataset_file,
        caption_column=caption_column,
        media_column=media_column,
        lora_trigger=lora_trigger,
        remove_llm_prefixes=remove_llm_prefixes,
    )
    logger.info(f"Loaded {len(dataset):,} captions")

    trust_manifest = os.environ.get("GOPEX_CONDITIONS_TRUST_MANIFEST", "0").strip().lower() in ("1", "true", "yes")
    prefilter = os.environ.get("GOPEX_CONDITIONS_PREFILTER", "0").strip().lower() in ("1", "true", "yes")
    if trust_manifest and require_latents and not prefilter:
        logger.info(
            "Caption shard filter: trusting manifest for latent coverage "
            "(GOPEX_CONDITIONS_TRUST_MANIFEST=1); still requiring latents in embed loop"
        )
    require_latents_at_runtime = require_latents

    if prefilter and (skip_existing or require_latents):
        kept, skipped_existing, skipped_no_latent = dataset.filter_shards(
            conditions_dir=output_path,
            latents_dir=latents_root,
            skip_existing=skip_existing,
            require_latents=require_latents,
        )
        parts = [f"{kept:,} to embed"]
        if skip_existing:
            parts.append(f"{skipped_existing:,} already on disk")
        if require_latents:
            parts.append(f"{skipped_no_latent:,} without latents")
        logger.info("Caption shard filter: %s", "; ".join(parts))
        if kept == 0:
            logger.info("No caption shards left to embed.")
            return
        skip_existing_at_runtime = False
        require_latents_at_runtime = False
    else:
        skip_existing_at_runtime = skip_existing
        if skip_existing or require_latents_at_runtime:
            logger.info(
                "Caption shard filter: deferred to embed loop (skip_existing=%s require_latents=%s)",
                skip_existing_at_runtime,
                require_latents_at_runtime,
            )

    apply_nvml_safe_cuda_patches()
    proc_device = _resolve_embeddings_processor_device(
        device=device,
        embeddings_device=embeddings_device,
        flat_dim_bridge_rank=flat_dim_bridge_rank,
        load_in_8bit=load_in_8bit,
    )

    # Load text encoder and embeddings processor
    with console.status("[bold]Loading Gemma text encoder...", spinner="dots"):
        text_encoder = load_text_encoder(
            text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
        )
        embeddings_processor = load_embeddings_processor(
            model_path,
            device=proc_device,
            dtype=torch.bfloat16,
            gemma_model_path=text_encoder_path,
            flat_dim_bridge_rank=flat_dim_bridge_rank,
        )

    logger.info("Text encoder and embeddings processor loaded successfully")

    # Persist bridge + aggregate weights so inference matches this preprocess run (not random init).
    text_stack_path = output_path / "text_stack_weights_preprocess.safetensors"
    if not text_stack_path.is_file():
        from safetensors.torch import save_file

        from ltx_trainer.preprocess_meta import write_preprocess_meta
        from ltx_trainer.text_stack_utils import feature_extractor_state_dict

        fe = embeddings_processor.feature_extractor
        if fe is not None:
            te_sd = feature_extractor_state_dict(fe, prefix="")
            save_file(te_sd, str(text_stack_path))
            logger.info("Saved preprocess text stack (bridge + aggregates): %s", text_stack_path)
            pre_root = output_path.parent if output_path.name == "conditions" else output_path
            from ltx_trainer.vcap.captioning import merge_preprocess_extra

            extra_meta = merge_preprocess_extra({"text_stack_path": str(text_stack_path.resolve())})
            write_preprocess_meta(
                pre_root,
                model_path=model_path,
                text_encoder_path=text_encoder_path,
                flat_dim_bridge_rank=flat_dim_bridge_rank,
                dataset_file=dataset_file,
                extra=extra_meta,
            )

    # Batched Gemma encode (tokenizer pads to max_length; see base_encoder.encode).
    embed_batch = int(os.environ.get("GOPEX_GEMMA_EMBED_BATCH", str(batch_size)))
    if embed_batch < 1:
        embed_batch = 1
    if embed_batch != batch_size:
        logger.info("GOPEX_GEMMA_EMBED_BATCH=%s — using batch_size=%s for caption embed", embed_batch, embed_batch)
        batch_size = embed_batch

    # Create dataloader (num_workers=0: Gemma tokenizer is not fork-safe; Py3.14 mp leaks semaphores)
    dl_workers = int(os.environ.get("GOPEX_CAPTIONS_NUM_WORKERS", "0"))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=dl_workers)

    # Process batches
    total_batches = len(dataloader)
    logger.info(f"Processing captions in {total_batches:,} batches...")

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
        task = progress.add_task("Processing captions", total=len(dataloader))
        embedded = 0
        for batch in dataloader:
            # Encode prompts using text_encoder.encode() + feature_extractor
            # (returns video/audio features before connector).
            # The connector is applied during training via embeddings_processor
            with torch.inference_mode():
                pending_idx: list[int] = []
                pending_prompts: list[str] = []
                pending_outputs: list[Path] = []
                for i in range(len(batch["prompt"])):
                    output_rel_path = Path(batch["output_path"][i])
                    output_file = output_path / output_rel_path

                    if skip_existing_at_runtime and output_file.is_file():
                        progress.advance(task)
                        continue
                    if require_latents_at_runtime and latents_root is not None:
                        if not (latents_root / output_rel_path).is_file():
                            progress.advance(task)
                            continue

                    pending_idx.append(i)
                    pending_prompts.append(batch["prompt"][i])
                    pending_outputs.append(output_file)

                if not pending_prompts:
                    continue

                encoded_list = text_encoder.encode(pending_prompts, padding_side="left")
                fe = embeddings_processor.feature_extractor
                fe_device = next(fe.parameters()).device

                if len(pending_prompts) == 1:
                    hidden_states, prompt_attention_mask = encoded_list[0]
                    if isinstance(hidden_states, (list, tuple)):
                        hidden_states = tuple(h.to(fe_device) for h in hidden_states)
                    else:
                        hidden_states = hidden_states.to(fe_device)
                    mask = prompt_attention_mask.to(fe_device)
                    video_prompt_embeds, audio_prompt_embeds = fe(hidden_states, mask, "left")
                    batch_video = [video_prompt_embeds[0]]
                    batch_audio = [audio_prompt_embeds[0] if audio_prompt_embeds is not None else None]
                    batch_masks = [prompt_attention_mask[0]]
                else:
                    layer_count = len(encoded_list[0][0])
                    layer_tensors: list[torch.Tensor] = []
                    for layer_i in range(layer_count):
                        layer_tensors.append(
                            torch.cat([encoded_list[j][0][layer_i] for j in range(len(encoded_list))], dim=0).to(
                                fe_device
                            )
                        )
                    batch_masks = torch.cat([encoded_list[j][1] for j in range(len(encoded_list))], dim=0).to(
                        fe_device
                    )
                    video_prompt_embeds, audio_prompt_embeds = fe(tuple(layer_tensors), batch_masks, "left")
                    batch_video = list(video_prompt_embeds)
                    batch_audio = (
                        list(audio_prompt_embeds) if audio_prompt_embeds is not None else [None] * len(pending_idx)
                    )
                    batch_masks = [batch_masks[j] for j in range(len(pending_idx))]

                for j, i in enumerate(pending_idx):
                    output_file = pending_outputs[j]
                    embedding_data = {
                        "video_prompt_embeds": batch_video[j].cpu().contiguous(),
                        "prompt_attention_mask": batch_masks[j].cpu().contiguous(),
                    }
                    if batch_audio[j] is not None:
                        embedding_data["audio_prompt_embeds"] = batch_audio[j].cpu().contiguous()

                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(embedding_data, output_file)
                    embedded += 1
                    if embedded % 50 == 0:
                        logger.info("Embedded %s caption shards (latest: %s)", embedded, output_file.relative_to(output_path))

            progress.advance(task)

    logger.info("Processed %s captions. Embeddings saved to %s", f"{len(dataset):,}", output_path)


@app.command()
def main(  # noqa: PLR0913
    dataset_file: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing captions and media paths",
    ),
    output_dir: str = typer.Option(
        ...,
        help="Output directory to save text embeddings",
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
    media_column: str = typer.Option(
        default="media_path",
        help="Column name in the dataset JSON/JSONL/CSV file containing media paths "
        "(used for output file naming and folder structure)",
    ),
    batch_size: int = typer.Option(
        default=8,
        help="Batch size for processing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    lora_trigger: str | None = typer.Option(
        default=None,
        help="Optional trigger word to prepend to each caption (activates the LoRA during inference)",
    ),
    remove_llm_prefixes: bool = typer.Option(
        default=False,
        help="Remove common LLM-generated prefixes from captions",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the Gemma text encoder in 8-bit precision to save GPU memory (requires bitsandbytes)",
    ),
    flat_dim_bridge_rank: int | None = typer.Option(
        default=None,
        help="If Gemma stacked width mismatches the LTX checkpoint, use low-rank experimental bridge with this rank "
        "(must match training config; defaults to rank 32 when omitted on mismatch)",
    ),
    skip_existing: bool = typer.Option(
        default=False,
        help="Skip items whose output .pt already exists (resume interrupted caption runs)",
    ),
) -> None:
    """Process text captions and save embeddings for video generation training.
    This script processes captions from metadata files and saves text embeddings
    that can be used for training video generation models. The output embeddings
    will maintain the same folder structure and naming as the corresponding media files.
    Note: This script is designed for LTX-2 models which use the Gemma text encoder.
    Examples:
        # Process captions with LTX-2 model
        python scripts/process_captions.py dataset.json --output-dir ./embeddings \\
            --model-path /path/to/ltx2_checkpoint.safetensors \\
            --text-encoder-path /path/to/gemma
        # Add a trigger word for LoRA training
        python scripts/process_captions.py dataset.json --output-dir ./embeddings \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --lora-trigger "mytoken"
        # Remove LLM-generated prefixes from captions
        python scripts/process_captions.py dataset.json --output-dir ./embeddings \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --remove-llm-prefixes
    """

    # Validate dataset file
    if not Path(dataset_file).is_file():
        raise typer.BadParameter(f"Dataset file not found: {dataset_file}")

    if lora_trigger:
        logger.info(f'LoRA trigger word "{lora_trigger}" will be prepended to all captions')

    # Process embeddings
    compute_captions_embeddings(
        dataset_file=dataset_file,
        output_dir=output_dir,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        caption_column=caption_column,
        media_column=media_column,
        lora_trigger=lora_trigger,
        remove_llm_prefixes=remove_llm_prefixes,
        batch_size=batch_size,
        device=device,
        load_in_8bit=load_text_encoder_in_8bit,
        flat_dim_bridge_rank=flat_dim_bridge_rank,
        skip_existing=skip_existing,
    )


if __name__ == "__main__":
    app()
