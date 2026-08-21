import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import wandb
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import functional as F  # noqa: N812

from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.config_display import print_config
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.gpu_utils import free_gpu_memory, free_gpu_memory_context, get_gpu_memory_gb
from ltx_trainer.hf_hub_utils import push_to_hub
from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder
from ltx_trainer.nvml_safe_cuda import (
    embeddings_processor_device,
    move_gemma_encode_outputs_to_device,
    resolve_gopex_cuda_device,
)
from ltx_trainer.model_loader import load_model as load_ltx_model
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.sigma_tracker import SigmaBucketTracker
from ltx_trainer.timestep_samplers import SAMPLERS
from ltx_trainer.training_state import ConfigFingerprint, RngStates, TrainingState
from ltx_trainer.training_strategies import get_training_strategy, strategy_requires_audio
from ltx_trainer.media_formats import is_still_image_path
from ltx_trainer.utils import open_image_as_srgb, save_image
from ltx_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import read_video, save_video

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

MEMORY_CHECK_INTERVAL = 200


def _gemma_encode_cpu_requested() -> bool:
    return os.environ.get("GOPEX_GEMMA_ENCODE_CPU", "").strip().lower() in ("1", "true", "yes")


def _resolve_gemma_load_device(freeze_dit: bool, startup_device: str) -> str:
    """Where to load Gemma for live caption training (second GPU preferred when frozen DiT)."""
    from ltx_trainer.text_stack_utils import gemma_caption_cache_enabled

    if _gemma_encode_cpu_requested():
        return "cpu"
    # 31B 8-bit often OOMs on 24GB cards; warm cache on CPU then unload (see GOPEX_GEMMA_UNLOAD_AFTER_CACHE).
    if freeze_dit and gemma_caption_cache_enabled():
        return "cpu"
    if freeze_dit and torch.cuda.is_available():
        train_raw = os.environ.get("GOPEX_TRAIN_CUDA_DEVICE", "0").strip() or "0"
        gemma_raw = os.environ.get("GOPEX_GEMMA_CUDA_DEVICE", "").strip()
        if gemma_raw:
            return resolve_gopex_cuda_device("GOPEX_GEMMA_CUDA_DEVICE", "cuda:1")
        if torch.cuda.device_count() > 1 and train_raw in ("0", ""):
            return "cuda:1"
    return startup_device


def _text_embed_sidecar_path(main_weights_path: Path) -> Path:
    """``lora_weights_step_00001.safetensors`` → ``text_embeds_weights_step_00001.safetensors``."""
    name = main_weights_path.name
    marker = "weights_step_"
    if marker in name:
        idx = name.index(marker)
        return main_weights_path.with_name(f"text_embeds_{name[idx:]}")
    return main_weights_path.with_name(f"text_embeds_{name}")


def _text_stack_sidecar_path(main_weights_path: Path) -> Path:
    """``lora_weights_step_00001.safetensors`` → ``text_stack_weights_step_00001.safetensors``."""
    name = main_weights_path.name
    marker = "weights_step_"
    if marker in name:
        idx = name.index(marker)
        return main_weights_path.with_name(f"text_stack_{name[idx:]}")
    return main_weights_path.with_name(f"text_stack_{name}")


def _text_connector_state_dict_for_save(embeddings_processor: EmbeddingsProcessor) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for k, v in embeddings_processor.video_connector.state_dict().items():
        out[f"video_connector.{k}"] = v.detach()
    if embeddings_processor.audio_connector is not None:
        for k, v in embeddings_processor.audio_connector.state_dict().items():
            out[f"audio_connector.{k}"] = v.detach()
    return out


def _load_text_connector_sidecar(embeddings_processor: EmbeddingsProcessor, path: Path) -> None:
    sd = load_file(path)
    v_sd = {k[len("video_connector.") :]: v for k, v in sd.items() if k.startswith("video_connector.")}
    try:
        embeddings_processor.video_connector.load_state_dict(v_sd, strict=True)
    except RuntimeError as exc:
        logger.warning(
            "Video connector sidecar shape mismatch (%s) — leaving connectors at init weights",
            exc,
        )
        return
    a_sd = {k[len("audio_connector.") :]: v for k, v in sd.items() if k.startswith("audio_connector.")}
    if a_sd:
        if embeddings_processor.audio_connector is None:
            logger.warning("Text-embed sidecar has audio_connector.* keys but no audio_connector on processor; skipped")
        else:
            try:
                embeddings_processor.audio_connector.load_state_dict(a_sd, strict=True)
            except RuntimeError as exc:
                logger.warning(
                    "Audio connector sidecar shape mismatch (%s) — video connector loaded; audio left at init",
                    exc,
                )


def _save_text_stack_sidecar(embeddings_processor: EmbeddingsProcessor, path: Path, save_dtype: torch.dtype) -> None:
    from ltx_trainer.text_stack_utils import feature_extractor_state_dict

    if embeddings_processor.feature_extractor is None:
        return
    te_sd = feature_extractor_state_dict(embeddings_processor.feature_extractor)
    te_sd = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in te_sd.items()}
    save_file(te_sd, path)


def _load_text_stack_sidecar(embeddings_processor: EmbeddingsProcessor, path: Path) -> None:
    from ltx_trainer.text_stack_utils import load_feature_extractor_state_dict

    if embeddings_processor.feature_extractor is None:
        logger.warning("Cannot load text_stack sidecar: feature_extractor is None")
        return
    load_feature_extractor_state_dict(embeddings_processor.feature_extractor, load_file(path))


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


@dataclass(frozen=True)
class TrainingStepOutput:
    """Output from a single training step."""

    loss: Tensor  # [B,] per-element loss (unreduced)
    sigma: Tensor  # [B,] sampled sigma, detached from computational graph
    latenthdr_loss: Tensor | None = None  # scalar L_ev when latenthdr.enabled
    av_fold_metrics: dict[str, float] | None = None


class LtxvTrainer:
    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        self._config = trainer_config
        self._exposure_head = None
        if IS_MAIN_PROCESS:
            print_config(trainer_config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        self._text_encoder = None
        self._gemma_encode_device = "cuda"
        self._gemma_caption_cache_dir: Path | None = None
        self._caption_index: dict[str, str] | None = None
        self._cached_validation_embeddings = self._load_text_encoder_and_cache_embeddings()
        self._load_models()
        self._setup_accelerator()
        freeze_dit = bool(self._config.model.finetune_text_stack and self._config.model.text_stack_freeze_dit)
        if self._config.model.training_mode == "lora" and not freeze_dit:
            self._setup_lora()
        elif freeze_dit:
            logger.info("text_stack_freeze_dit: skipping LoRA adapters on frozen DiT (saves VRAM)")
        self._loaded_checkpoint_path: Path | None = None
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._collect_trainable_params()
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths: list[Path] = []
        self._training_state_paths: list[Path] = []
        self._training_state_size_warned = False
        self._wandb_run = None
        self._sigma_tracker = SigmaBucketTracker()
        self._av_fold_sidecar_warned = False

        if self._config.model.finetune_text_connectors and self._config.validation.prompts:
            if self._cached_validation_embeddings and not all(
                e.features_pre_connector for e in self._cached_validation_embeddings
            ):
                logger.warning(
                    "Validation cache is post-connector; re-run with finetune_text_connectors to use pre-connector cache."
                )
            elif self._cached_validation_embeddings:
                logger.info(
                    "finetune_text_connectors: validation uses pre-connector features + live connectors each sample."
                )

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """
        Start the training process.
        Args:
            disable_progress_bars: Disable Rich progress bars (useful for multi-process runs).
            step_callback: Optional callback invoked after each optimization step.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        initial_step, training_state = self._resume_state
        resuming = training_state is not None

        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._init_optimizer()

        if training_state is not None and not self._restore_training_state(training_state):
            initial_step = 0
            resuming = False

        # Initialize W&B after restore so we only resume the run when state restore succeeds.
        resume_run_id = training_state.wandb_run_id if resuming and training_state is not None else None
        self._init_wandb(resume_run_id=resume_run_id)

        self._init_dataloader()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()
        self._maybe_warm_gemma_caption_cache()

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save the training configuration as YAML
        self._save_config()

        remaining_steps = cfg.optimization.steps - initial_step
        if remaining_steps <= 0:
            raise ValueError(
                f"No remaining training steps: initial_step={initial_step} >= "
                f"target_steps={cfg.optimization.steps}. Nothing to train."
            )

        if resuming:
            logger.info(f"🚀 Resuming training from step {initial_step} → {cfg.optimization.steps}")
        else:
            logger.info("🚀 Starting training...")

        # Create progress tracking (disabled for non-main processes or when explicitly disabled)
        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=remaining_steps,
        )

        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning("Progress bars disabled. Intermediate status messages will be logged instead.")

        self._transformer.train()
        self._global_step = initial_step

        peak_mem_during_training = start_mem

        sampled_videos_paths = None

        with progress:
            if cfg.validation.interval and not cfg.validation.skip_initial_validation:
                sampled_videos_paths = self._sample_videos(progress)
                if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                    self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

            self._accelerator.wait_for_everyone()

            for step in range(remaining_steps * cfg.optimization.gradient_accumulation_steps):
                # Get next batch, reset the dataloader if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (step + 1) % cfg.optimization.gradient_accumulation_steps == 0
                    micro_idx = (step % cfg.optimization.gradient_accumulation_steps) + 1
                    accum = cfg.optimization.gradient_accumulation_steps
                    if is_optimization_step:
                        self._global_step += 1

                    log_every = max(1, int(os.environ.get("GOPEX_TRAIN_LOG_EVERY", "1")))
                    if (
                        IS_MAIN_PROCESS
                        and is_optimization_step
                        and (log_every == 1 or self._global_step % log_every == 0)
                    ):
                        logger.info(
                            "Step %s/%s starting (grad-accum micro-batch %s/%s, device=%s)",
                            self._global_step,
                            cfg.optimization.steps,
                            micro_idx,
                            accum,
                            self._accelerator.device,
                        )

                    output = self._training_step(batch)
                    self._accelerator.backward(output.loss.mean())

                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # Save checkpoint before validation so a decode/sample crash
                    # does not throw away an interval of training progress.
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                    ):
                        self._save_checkpoint()

                    # Run validation if needed
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                    ):
                        if self._accelerator.distributed_type == DistributedType.FSDP:
                            # FSDP: All processes must participate in validation
                            sampled_videos_paths = self._sample_videos(progress)
                            if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)
                        # DDP: Only main process runs validation
                        elif IS_MAIN_PROCESS:
                            sampled_videos_paths = self._sample_videos(progress)
                            if sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths)

                    self._accelerator.wait_for_everyone()

                    # Update progress and log metrics
                    current_lr = self._optimizer.param_groups[0]["lr"]
                    step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps
                    step_loss = output.loss.detach().mean().item()

                    progress.update_training(
                        loss=step_loss,
                        lr=current_lr,
                        step_time=step_time,
                        advance=is_optimization_step,
                    )

                    # Log metrics to W&B (only on main process and optimization steps)
                    if IS_MAIN_PROCESS and is_optimization_step:
                        # Track per-element loss by sigma bucket
                        self._sigma_tracker.update(output.sigma.cpu().tolist(), output.loss.detach().cpu().tolist())
                        metrics = {
                            "train/loss": step_loss,
                            "train/learning_rate": current_lr,
                            "train/step_time": step_time,
                            "train/global_step": self._global_step,
                        }
                        if output.latenthdr_loss is not None:
                            metrics["train/latenthdr_ev_loss"] = float(output.latenthdr_loss.detach().item())
                        if output.av_fold_metrics:
                            metrics.update({f"train/{k}": v for k, v in output.av_fold_metrics.items()})
                        metrics.update(self._sigma_tracker.get_metrics())
                        self._log_metrics(metrics)

                    # Line-logged progress (tee/log files; independent of Rich progress bar)
                    if IS_MAIN_PROCESS and is_optimization_step:
                        log_every = max(1, int(os.environ.get("GOPEX_TRAIN_LOG_EVERY", "10")))
                        if self._global_step % log_every == 0:
                            elapsed = time.time() - train_start_time
                            steps_done = max(1, self._global_step - initial_step)
                            eta_s = elapsed / steps_done * max(0, remaining_steps - steps_done)
                            fold_suffix = ""
                            if output.av_fold_metrics:
                                parts = [
                                    f"{k.split('/')[-1]}={v:.3f}"
                                    for k, v in sorted(output.av_fold_metrics.items())
                                ]
                                fold_suffix = " | " + " ".join(parts)
                            logger.info(
                                "Step %s/%s — loss=%.4f lr=%.2e %.2fs/step ETA~%s%s",
                                self._global_step,
                                cfg.optimization.steps,
                                step_loss,
                                current_lr,
                                step_time,
                                f"{int(eta_s // 3600)}h{int((eta_s % 3600) // 60)}m",
                                fold_suffix,
                            )

                    # Fallback logging when progress bars are disabled
                    if disable_progress_bars and IS_MAIN_PROCESS and self._global_step % 20 == 0:
                        elapsed = time.time() - train_start_time
                        steps_done = self._global_step - initial_step
                        if steps_done > 0:
                            total_estimated = elapsed / steps_done * remaining_steps
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."
                        logger.info(
                            f"Step {self._global_step}/{cfg.optimization.steps} - "
                            f"Loss: {step_loss:.4f}, LR: {current_lr:.2e}, "
                            f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                        )

                    # Sample GPU memory periodically
                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Calculate steps/second over entire training
        total_time_seconds = train_end_time - train_start_time
        steps_per_second = remaining_steps / total_time_seconds

        samples_per_second = steps_per_second * self._accelerator.num_processes * cfg.optimization.batch_size

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=cfg.optimization.batch_size * self._accelerator.num_processes,
        )

        saved_path = self._save_checkpoint()

        if IS_MAIN_PROCESS:
            # Log the training statistics
            self._log_training_stats(stats)

            # Upload artifacts to hub if enabled
            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            # Log final stats to W&B
            if self._wandb_run is not None:
                self._log_metrics(
                    {
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }
                )
                self._wandb_run.finish()

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()

        return saved_path, stats

    def _connector_device(self) -> torch.device:
        """DiT stays on the Accelerate device; optional ``GOPEX_CONNECTOR_CUDA_DEVICE`` offloads connectors."""
        train_dev = self._accelerator.device
        if not torch.cuda.is_available() or self._embeddings_processor is None:
            return train_dev
        conn = resolve_gopex_cuda_device("GOPEX_CONNECTOR_CUDA_DEVICE", train_dev)
        return torch.device(conn)

    def _place_embeddings_processor(self) -> None:
        """Move text connectors (and optional feature extractor) to the training connector GPU."""
        if self._embeddings_processor is None or not torch.cuda.is_available():
            return
        conn_dev = self._connector_device()
        self._embeddings_processor.video_connector.to(conn_dev)
        if self._embeddings_processor.audio_connector is not None:
            self._embeddings_processor.audio_connector.to(conn_dev)
        fe = self._embeddings_processor.feature_extractor
        if fe is not None:
            fe.to(conn_dev)

    def _apply_text_connectors(
        self,
        video_features: Tensor,
        audio_features: Tensor | None,
        prompt_mask: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Run frozen/trainable connectors; handles cross-GPU when ``GOPEX_CONNECTOR_CUDA_DEVICE`` is set."""
        train_dev = self._accelerator.device
        conn_dev = self._connector_device()
        additive_mask = convert_to_additive_mask(prompt_mask, video_features.dtype)
        if conn_dev != train_dev:
            video_features = video_features.to(conn_dev)
            if audio_features is not None:
                audio_features = audio_features.to(conn_dev)
            additive_mask = additive_mask.to(conn_dev)
        video_embeds, audio_embeds, attention_mask = self._embeddings_processor.create_embeddings(
            video_features, audio_features, additive_mask
        )
        if conn_dev != train_dev:
            video_embeds = video_embeds.to(train_dev)
            if audio_embeds is not None:
                audio_embeds = audio_embeds.to(train_dev)
            attention_mask = attention_mask.to(train_dev)
        return video_embeds, audio_embeds, attention_mask

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> TrainingStepOutput:
        """Perform a single training step using the configured strategy."""
        conditions = batch["conditions"]

        if self._config.model.finetune_text_stack and self._config.model.text_stack_live_captions:
            video_embeds, audio_embeds, attention_mask = self._encode_live_captions(batch)
            conditions["video_prompt_embeds"] = video_embeds
            conditions["audio_prompt_embeds"] = audio_embeds
            conditions["prompt_attention_mask"] = attention_mask
        else:
            if "video_prompt_embeds" in conditions:
                video_features = conditions["video_prompt_embeds"]
                audio_features = conditions.get("audio_prompt_embeds")
            else:
                video_features = conditions["prompt_embeds"]
                audio_features = conditions["prompt_embeds"]

            mask = conditions["prompt_attention_mask"]
            video_embeds, audio_embeds, attention_mask = self._apply_text_connectors(
                video_features, audio_features, mask
            )
            conditions["video_prompt_embeds"] = video_embeds
            conditions["audio_prompt_embeds"] = audio_embeds
            conditions["prompt_attention_mask"] = attention_mask

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        model_inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)

        # Run transformer forward pass with Modality-based interface
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)
        if model_inputs.video is not None and model_inputs.video.enabled:
            sigma = model_inputs.video.sigma.detach()
        elif model_inputs.audio is not None:
            sigma = model_inputs.audio.sigma.detach()
        else:
            raise RuntimeError("Training step produced no enabled video or audio modality")

        av_fold_metrics: dict[str, float] | None = None
        if self._config.av_fold.enabled:
            from ltx_trainer.av_fold_training import apply_av_fold_training

            with_audio = bool(getattr(self._training_strategy.config, "with_audio", False))
            loss, av_fold_metrics = apply_av_fold_training(
                loss,
                batch,
                cfg=self._config.av_fold,
                with_audio=with_audio,
            )
            if IS_MAIN_PROCESS and not self._av_fold_sidecar_warned:
                from ltx_trainer.av_fold_training import sidecar_coverage_summary

                cov = sidecar_coverage_summary(batch)
                if not any(cov.values()):
                    allow_empty = os.environ.get("GOPEX_AV_FOLD_ALLOW_EMPTY", "").strip().lower() in (
                        "1",
                        "true",
                        "yes",
                    )
                    require = bool(getattr(self._config.av_fold, "require_sidecars", True))
                    msg = (
                        "av_fold.enabled but no fold sidecars in batch — re-encode/backfill with "
                        "GOPEX_ENABLE_AV_FOLD=1 (see documents/LTX_FOLD_HOOKS.md), or set "
                        "GOPEX_AV_FOLD_ALLOW_EMPTY=1 / av_fold.require_sidecars=false"
                    )
                    if require and not allow_empty:
                        raise RuntimeError(msg)
                    logger.warning("%s. Training continues unweighted.", msg)
                    self._av_fold_sidecar_warned = True
                elif av_fold_metrics:
                    self._av_fold_sidecar_warned = True

        latenthdr_loss: Tensor | None = None
        if self._config.latenthdr.enabled and self._exposure_head is not None:
            from ltx_trainer.latenthdr import compute_latenthdr_ev_loss

            latenthdr_loss = compute_latenthdr_ev_loss(
                batch,
                self._exposure_head,
                device=self._accelerator.device,
            )
            if latenthdr_loss is not None:
                loss = loss + self._config.latenthdr.lambda_ev * latenthdr_loss

        return TrainingStepOutput(
            loss=loss,
            sigma=sigma,
            latenthdr_loss=latenthdr_loss,
            av_fold_metrics=av_fold_metrics,
        )

    def _encode_live_captions(self, batch: dict) -> tuple[Tensor, Tensor | None, Tensor]:
        """Gemma encode → feature_extractor → connectors for each sample in the batch."""
        raw = batch.get("caption")
        if raw is None:
            raise KeyError("Batch missing 'caption'; set data.dataset_manifest_path for live text-stack training")
        captions: list[str] = [raw] if isinstance(raw, str) else list(raw)

        device = self._accelerator.device
        video_parts: list[Tensor] = []
        audio_parts: list[Tensor] = []
        mask_parts: list[Tensor] = []

        proc_device = str(self._accelerator.device)
        encode_dev = self._gemma_encode_device
        cache_dir = self._gemma_caption_cache_dir
        from ltx_trainer.text_stack_utils import (
            gemma_caption_cache_enabled,
            load_gemma_caption_cache,
            save_gemma_caption_cache,
        )

        cache_only = (
            self._text_encoder is None
            and cache_dir is not None
            and gemma_caption_cache_enabled()
        )
        if self._text_encoder is None and not cache_only:
            raise RuntimeError(
                "Live caption training requires text_encoder or a populated gemma_caption_cache "
                "(finetune_text_stack + text_stack_live_captions)"
            )

        for caption in captions:
            hidden_states: tuple[Tensor, ...] | Tensor
            prompt_mask: Tensor
            cached = None
            if cache_dir is not None and gemma_caption_cache_enabled():
                cached = load_gemma_caption_cache(cache_dir, caption)
            if cached is not None:
                hidden_states, prompt_mask = move_gemma_encode_outputs_to_device(
                    cached[0], cached[1], proc_device
                )
            else:
                if self._text_encoder is None:
                    raise RuntimeError(
                        f"Gemma caption cache miss and text encoder unloaded (caption hash "
                        f"{caption[:48]!r}…). Re-run with GOPEX_GEMMA_UNLOAD_AFTER_CACHE=0 or delete "
                        f"{cache_dir} and restart to re-warm."
                    )
                encoded = self._text_encoder.encode([caption], padding_side="left")
                hidden_states, prompt_mask = encoded[0] if isinstance(encoded, list) else encoded
                if cache_dir is not None and gemma_caption_cache_enabled():
                    save_gemma_caption_cache(cache_dir, caption, hidden_states, prompt_mask)
                hidden_states, prompt_mask = move_gemma_encode_outputs_to_device(
                    hidden_states, prompt_mask, encode_dev
                )
                if encode_dev != proc_device:
                    hidden_states, prompt_mask = move_gemma_encode_outputs_to_device(
                        hidden_states, prompt_mask, proc_device
                    )
            out = self._embeddings_processor.process_hidden_states(hidden_states, prompt_mask, "left")
            video_parts.append(out.video_encoding)
            mask_parts.append(out.attention_mask)
            if out.audio_encoding is not None:
                audio_parts.append(out.audio_encoding)

        video_embeds = torch.cat(video_parts, dim=0).to(device)
        attention_mask = torch.cat(mask_parts, dim=0).to(device)
        audio_embeds = torch.cat(audio_parts, dim=0).to(device) if audio_parts else None
        return video_embeds, audio_embeds, attention_mask

    def _maybe_warm_gemma_caption_cache(self) -> None:
        """One-time Gemma encode per unique manifest caption (disk cache for training steps)."""
        if self._text_encoder is None or not self._caption_index:
            return
        from ltx_trainer.text_stack_utils import (
            gemma_caption_cache_enabled,
            gemma_caption_cache_path,
            save_gemma_caption_cache,
        )

        if not gemma_caption_cache_enabled():
            return
        cache_dir = self._gemma_caption_cache_dir or (Path(self._config.output_dir) / "gemma_caption_cache")
        self._gemma_caption_cache_dir = cache_dir
        unique = sorted(set(self._caption_index.values()))
        missing = [c for c in unique if not gemma_caption_cache_path(cache_dir, c).is_file()]
        if not missing:
            logger.info("Gemma caption cache: %s entries, all present under %s", len(unique), cache_dir)
            if os.environ.get("GOPEX_GEMMA_UNLOAD_AFTER_CACHE", "1").strip().lower() not in ("0", "false", "no"):
                self._unload_live_gemma_encoder()
            return
        logger.info(
            "Warming Gemma caption cache: %s new / %s unique captions → %s",
            len(missing),
            len(unique),
            cache_dir,
        )
        t0 = time.time()
        for i, caption in enumerate(missing, start=1):
            encoded = self._text_encoder.encode([caption], padding_side="left")
            hidden_states, prompt_mask = encoded[0] if isinstance(encoded, list) else encoded
            save_gemma_caption_cache(cache_dir, caption, hidden_states, prompt_mask)
            if i == 1 or i % 25 == 0 or i == len(missing):
                logger.info("Gemma cache warm %s/%s (%.0fs elapsed)", i, len(missing), time.time() - t0)
        logger.info("Gemma caption cache ready (%s files, %.0fs)", len(unique), time.time() - t0)
        if os.environ.get("GOPEX_GEMMA_UNLOAD_AFTER_CACHE", "1").strip().lower() not in ("0", "false", "no"):
            self._unload_live_gemma_encoder()

    def _unload_live_gemma_encoder(self) -> None:
        """Drop Gemma weights after caption cache warm (training uses disk cache only)."""
        if self._text_encoder is None:
            return
        import gc

        del self._text_encoder
        self._text_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded Gemma text encoder after caption cache warm (training uses cache only)")

    @free_gpu_memory_context(after=True)
    def _load_text_encoder_and_cache_embeddings(self) -> list[CachedPromptEmbeddings] | None:
        """Load text encoder + embeddings processor, compute and cache validation embeddings."""

        # This method:
        #   1. Loads the pure Gemma text encoder on GPU
        #   2. Loads the embeddings processor (feature extractor + connectors)
        #   3. If validation prompts are configured, computes and caches their embeddings
        #   4. Unloads the Gemma model entirely, keeps the embeddings processor for training

        startup_device = embeddings_processor_device("cuda")
        if startup_device == "cpu":
            logger.info(
                "NVML-safe mode: Gemma + embeddings processor on CPU for validation cache; "
                "connectors move to training GPU after model prep."
            )

        # Load text encoder (pure Gemma LLM)
        freeze_dit = bool(self._config.model.finetune_text_stack and self._config.model.text_stack_freeze_dit)
        gemma_device = _resolve_gemma_load_device(freeze_dit, startup_device)
        self._gemma_encode_device = gemma_device
        gemma_8bit = bool(self._config.acceleration.load_text_encoder_in_8bit) and gemma_device != "cpu"
        if freeze_dit:
            if gemma_device == "cpu":
                logger.info(
                    "text_stack_freeze_dit: Gemma bf16 on CPU (GOPEX_GEMMA_ENCODE_CPU); "
                    "int8 DiT + trainable stack on training GPU"
                )
            else:
                logger.info(
                    "text_stack_freeze_dit: Gemma %s on %s; int8 DiT + trainable stack on training GPU",
                    "8bit" if gemma_8bit else "bf16",
                    gemma_device,
                )
        logger.debug("Loading text encoder...")
        text_encoder = load_text_encoder(
            gemma_model_path=self._config.model.text_encoder_path,
            device=gemma_device,
            dtype=torch.bfloat16,
            load_in_8bit=gemma_8bit,
        )

        # Load embeddings processor (feature extractor + connectors)
        logger.debug("Loading embeddings processor...")
        self._embeddings_processor = load_embeddings_processor(
            checkpoint_path=self._config.model.model_path,
            device=startup_device,
            dtype=torch.bfloat16,
            gemma_model_path=self._config.model.text_encoder_path,
            gemma_encode_stack_dims=(
                dict(self._config.model.gemma_encode_stack_dims) if self._config.model.gemma_encode_stack_dims else None
            ),
            require_matched_gemma_text_flat_dim=self._config.model.require_matched_gemma_text_flat_dim,
            flat_dim_bridge_rank=self._config.model.flat_dim_bridge_rank,
        )

        # Cache validation embeddings if prompts are configured
        cached_embeddings = None
        if self._config.validation.prompts:
            logger.info(f"Pre-computing embeddings for {len(self._config.validation.prompts)} validation prompts...")
            cached_embeddings = []
            cache_pre_connector = bool(self._config.model.finetune_text_connectors)
            with torch.inference_mode():
                for prompt in self._config.validation.prompts:
                    pos_encoded = text_encoder.encode([prompt])
                    neg_encoded = text_encoder.encode([self._config.validation.negative_prompt])
                    pos_hs, pos_mask = pos_encoded[0] if isinstance(pos_encoded, list) else pos_encoded
                    neg_hs, neg_mask = neg_encoded[0] if isinstance(neg_encoded, list) else neg_encoded
                    pos_hs, pos_mask = move_gemma_encode_outputs_to_device(pos_hs, pos_mask, startup_device)
                    neg_hs, neg_mask = move_gemma_encode_outputs_to_device(neg_hs, neg_mask, startup_device)
                    if cache_pre_connector:
                        pos_v, pos_a = self._embeddings_processor.feature_extractor(pos_hs, pos_mask, "left")
                        neg_v, neg_a = self._embeddings_processor.feature_extractor(neg_hs, neg_mask, "left")
                        cached_embeddings.append(
                            CachedPromptEmbeddings(
                                video_context_positive=pos_v.cpu(),
                                audio_context_positive=pos_a.cpu() if pos_a is not None else None,
                                video_context_negative=neg_v.cpu(),
                                audio_context_negative=neg_a.cpu() if neg_a is not None else None,
                                prompt_attention_mask_positive=pos_mask.cpu(),
                                prompt_attention_mask_negative=neg_mask.cpu(),
                                features_pre_connector=True,
                            )
                        )
                    else:
                        pos_out = self._embeddings_processor.process_hidden_states(pos_hs, pos_mask)
                        neg_out = self._embeddings_processor.process_hidden_states(neg_hs, neg_mask)
                        cached_embeddings.append(
                            CachedPromptEmbeddings(
                                video_context_positive=pos_out.video_encoding.cpu(),
                                audio_context_positive=(
                                    pos_out.audio_encoding.cpu() if pos_out.audio_encoding is not None else None
                                ),
                                video_context_negative=neg_out.video_encoding.cpu(),
                                audio_context_negative=(
                                    neg_out.audio_encoding.cpu() if neg_out.audio_encoding is not None else None
                                ),
                            )
                        )

        keep_live = bool(self._config.model.finetune_text_stack and self._config.model.text_stack_live_captions)
        if keep_live:
            self._text_encoder = text_encoder
            if freeze_dit:
                logger.info(
                    "finetune_text_stack: Gemma on CPU for encode; feature_extractor + frozen DiT on training GPU"
                )
            else:
                logger.info(
                    "finetune_text_stack: keeping Gemma text encoder and feature_extractor on GPU for live captions"
                )
        else:
            del text_encoder

        if not self._config.model.finetune_text_stack:
            self._embeddings_processor.feature_extractor = None
            logger.debug("Validation prompt embeddings cached. Gemma model unloaded; feature_extractor dropped")
        elif not keep_live:
            del text_encoder
            logger.debug("Text stack training without live captions uses precomputed features only")

        return cached_embeddings

    def _load_models(self) -> None:
        """Load the LTX-2 model components."""
        # Load audio components if:
        # 1. Training strategy requires audio (training the audio branch), OR
        # 2. Validation is configured to generate audio (even if not training audio)
        load_audio = strategy_requires_audio(self._config.training_strategy) or self._config.validation.generate_audio

        # Check if we need VAE encoder (for image or reference video conditioning)
        need_vae_encoder = (
            self._config.validation.images is not None or self._config.validation.reference_videos is not None
        )

        # Load all model components (except text encoder - already handled)
        components = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=need_vae_encoder,  # Needed for image conditioning
            with_video_vae_decoder=True,  # Needed for validation sampling
            with_audio_vae_decoder=load_audio,
            with_vocoder=load_audio,
            with_text_encoder=False,  # Text encoder handled separately
        )

        # Extract components
        self._transformer = components.transformer
        self._vae_decoder = components.video_vae_decoder.to(dtype=torch.bfloat16)
        self._vae_encoder = components.video_vae_encoder
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to(dtype=torch.bfloat16)
        self._scheduler = components.scheduler
        self._audio_vae = components.audio_vae_decoder
        self._vocoder = components.vocoder
        # Note: self._embeddings_processor was set in _load_text_encoder_and_cache_embeddings

        # Determine initial dtype based on training mode.
        # Note: For FSDP + LoRA, we'll cast to FP32 later in _prepare_models_for_training()
        # after the accelerator is set up, and we can detect FSDP.
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        self._transformer = self._transformer.to(dtype=transformer_dtype)

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.info(f'Quantizing model with "{self._config.acceleration.quantization}". This may take a while...')
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

        # Freeze all models. We later unfreeze the transformer based on training mode.
        # Note: embedding_connectors are already frozen (they come from the frozen text encoder)
        self._vae_decoder.requires_grad_(False)
        if self._vae_encoder is not None:
            self._vae_encoder.requires_grad_(False)
        self._transformer.requires_grad_(False)
        if self._audio_vae is not None:
            self._audio_vae.requires_grad_(False)
        if self._vocoder is not None:
            self._vocoder.requires_grad_(False)

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode."""
        freeze_dit = bool(self._config.model.finetune_text_stack and self._config.model.text_stack_freeze_dit)

        if freeze_dit:
            self._transformer.requires_grad_(False)
        elif self._config.model.training_mode == "lora":
            pass  # LoRA adapter attached before checkpoint load
        elif self._config.model.training_mode == "full":
            self._transformer.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        n_diffusion = sum(p.numel() for p in self._trainable_params)
        mode = self._config.model.training_mode

        if self._config.model.finetune_text_stack:
            fe = self._embeddings_processor.feature_extractor
            if fe is None:
                raise RuntimeError("finetune_text_stack requires feature_extractor on the embeddings processor")
            fe.requires_grad_(True)
            fe_params = [p for p in fe.parameters() if p.requires_grad]
            self._trainable_params.extend(fe_params)
            n_fe = sum(p.numel() for p in fe_params)
            logger.info("Trainable text stack (feature_extractor): %s parameters", f"{n_fe:,}")

        if self._config.model.finetune_text_connectors:
            self._embeddings_processor.video_connector.requires_grad_(True)
            if self._embeddings_processor.audio_connector is not None:
                self._embeddings_processor.audio_connector.requires_grad_(True)
            extra = [p for p in self._embeddings_processor.video_connector.parameters() if p.requires_grad]
            if self._embeddings_processor.audio_connector is not None:
                extra.extend([p for p in self._embeddings_processor.audio_connector.parameters() if p.requires_grad])
            self._trainable_params.extend(extra)
            n_conn = sum(p.numel() for p in extra)
            n_total = n_diffusion + n_conn
            logger.info(
                "Trainable parameters: %s (diffusion, %s) + %s (text embedding connectors) = %s total",
                f"{n_diffusion:,}",
                mode,
                f"{n_conn:,}",
                f"{n_total:,}",
            )
        else:
            self._embeddings_processor.video_connector.requires_grad_(False)
            if self._embeddings_processor.audio_connector is not None:
                self._embeddings_processor.audio_connector.requires_grad_(False)
            logger.info(
                "Trainable parameters: %s (%s)%s",
                f"{n_diffusion:,}",
                mode,
                " + LoRA adapters" if mode == "lora" else "",
            )

        self._setup_latenthdr()
        logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _setup_latenthdr(self) -> None:
        """Load optional LatentHDR exposure head for joint L_ev."""
        if not self._config.latenthdr.enabled:
            self._exposure_head = None
            return
        from ltx_trainer.latenthdr import FiLMResidualExposureHead, load_exposure_head_from_checkpoint

        cfg = self._config.latenthdr
        if cfg.exposure_head_checkpoint:
            self._exposure_head = load_exposure_head_from_checkpoint(
                cfg.exposure_head_checkpoint,
                device="cpu",
            )
        else:
            self._exposure_head = FiLMResidualExposureHead(latent_channels=cfg.latent_channels)

        if cfg.train_exposure_head:
            self._exposure_head.requires_grad_(True)
            head_params = [p for p in self._exposure_head.parameters() if p.requires_grad]
            self._trainable_params.extend(head_params)
            logger.info(
                "LatentHDR exposure head trainable: %s parameters (λ_ev=%s)",
                f"{sum(p.numel() for p in head_params):,}",
                cfg.lambda_ev,
            )
        else:
            self._exposure_head.requires_grad_(False)
            logger.info("LatentHDR exposure head frozen (λ_ev=%s)", cfg.lambda_ev)

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        # Wrap the transformer with PEFT to add LoRA layers
        # noinspection PyTypeChecker
        self._transformer = get_peft_model(self._transformer, lora_config)

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config, then resolve resume state."""
        if not self._config.model.load_checkpoint:
            self._resume_state: tuple[int, TrainingState | None] = (0, None)
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}")
            self._resume_state = (0, None)
            return

        self._loaded_checkpoint_path = checkpoint_path
        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        elif self._text_stack_freeze_dit_enabled() and "text_stack" in checkpoint_path.name:
            self._maybe_load_text_stack_checkpoint(checkpoint_path)
            logger.info("Loaded text stack checkpoint (frozen DiT — no LoRA weights)")
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

        self._resume_state = self._resolve_resume_state()

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint."""
        state_dict = load_file(checkpoint_path)
        self._transformer.load_state_dict(state_dict, strict=True)

        logger.info("✅ Full model checkpoint loaded successfully")
        self._maybe_load_text_embed_connector_checkpoint(checkpoint_path)
        self._maybe_load_text_stack_checkpoint(checkpoint_path)

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        state_dict = load_file(checkpoint_path)

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

        # Load LoRA weights and verify all weights were loaded
        base_model = self._transformer.get_base_model()
        set_peft_model_state_dict(base_model, state_dict)

        logger.info("✅ LoRA checkpoint loaded successfully")
        self._maybe_load_text_embed_connector_checkpoint(checkpoint_path)
        self._maybe_load_text_stack_checkpoint(checkpoint_path)

    def _maybe_load_text_stack_checkpoint(self, main_checkpoint: Path) -> None:
        if not self._config.model.finetune_text_stack:
            return
        sidecar = _text_stack_sidecar_path(main_checkpoint)
        if not sidecar.is_file():
            return
        _load_text_stack_sidecar(self._embeddings_processor, sidecar)
        logger.info("Loaded text stack weights from %s", sidecar)

    def _maybe_load_text_embed_connector_checkpoint(self, main_checkpoint: Path) -> None:
        sidecar = _text_embed_sidecar_path(main_checkpoint)
        if not sidecar.is_file():
            if self._config.model.finetune_text_connectors:
                logger.warning(
                    "finetune_text_connectors is enabled but no sidecar at %s — connectors stay at init weights",
                    sidecar,
                )
            return
        _load_text_connector_sidecar(self._embeddings_processor, sidecar)
        if self._config.model.finetune_text_connectors:
            logger.info("Loaded text connector weights from %s", sidecar)
        else:
            logger.info("Loaded frozen text connector weights from %s (LoRA-only training)", sidecar)

    def _resolve_resume_state(self) -> tuple[int, TrainingState | None]:
        """Determine resume state by looking for a training state file next to the loaded checkpoint.
        Returns (initial_step, TrainingState or None).
        If no_resume config is set, no checkpoint loaded, or no state file found: returns (0, None).
        """
        if self._config.checkpoints.no_resume or self._loaded_checkpoint_path is None:
            return 0, None

        state = self._load_training_state(self._loaded_checkpoint_path)
        if state is None:
            return 0, None

        fp = state.config_fingerprint
        cfg = self._config
        mismatches: list[str] = []
        if fp.optimizer_type != cfg.optimization.optimizer_type:
            mismatches.append(f"optimizer_type: {fp.optimizer_type} → {cfg.optimization.optimizer_type}")
        if fp.scheduler_type != cfg.optimization.scheduler_type:
            mismatches.append(f"scheduler_type: {fp.scheduler_type} → {cfg.optimization.scheduler_type}")
        if fp.training_mode != cfg.model.training_mode:
            mismatches.append(f"training_mode: {fp.training_mode} → {cfg.model.training_mode}")
        if (
            cfg.model.training_mode == "lora"
            and cfg.lora is not None
            and fp.lora_rank is not None
            and fp.lora_rank != cfg.lora.rank
        ):
            mismatches.append(f"lora_rank: {fp.lora_rank} → {cfg.lora.rank}")
        if bool(fp.finetune_text_connectors) != bool(cfg.model.finetune_text_connectors):
            mismatches.append(
                f"finetune_text_connectors: {fp.finetune_text_connectors} → {cfg.model.finetune_text_connectors}"
            )
        if bool(fp.finetune_text_stack) != bool(cfg.model.finetune_text_stack):
            mismatches.append(f"finetune_text_stack: {fp.finetune_text_stack} → {cfg.model.finetune_text_stack}")
        if mismatches:
            logger.warning(
                f"⚠️ Training state config mismatch ({', '.join(mismatches)}). "
                "Starting from step 0. Set checkpoints.no_resume=true to silence this warning."
            )
            return 0, None

        if state.global_step < 0:
            logger.warning(f"⚠️ Training state has invalid global_step={state.global_step!r}. Starting from step 0.")
            return 0, None
        logger.info(f"📌 Resuming from step {state.global_step}")
        return state.global_step, state

    @staticmethod
    def _load_training_state(checkpoint_path: Path) -> TrainingState | None:
        """Load training state file that corresponds to a checkpoint weights file."""
        match = re.search(r"step_(\d+)", checkpoint_path.name)
        if not match:
            return None

        step_str = match.group(1)
        state_path = checkpoint_path.parent / f"training_state_step_{step_str}.pt"

        if not state_path.exists():
            return None

        try:
            raw: dict = torch.load(state_path, map_location="cpu", weights_only=False)
            state = TrainingState.from_save_dict(raw)
            logger.info(f"📥 Loaded training state from {state_path}")
            return state
        except Exception as e:
            logger.warning(f"⚠️ Failed to load training state from {state_path}: {e}. Starting from step 0.")
            return None

    def _restore_training_state(self, training_state: TrainingState) -> bool:
        """Restore optimizer, scheduler, and RNG states from a loaded TrainingState.
        Must be called after _init_optimizer() (which calls accelerator.prepare).
        Returns True if restore succeeded, False if it failed (caller should fall back to step 0).
        """
        try:
            if training_state.optimizer_state_dict is not None:
                self._optimizer.load_state_dict(training_state.optimizer_state_dict)
                logger.debug("Restored optimizer state (full mode)")

            if training_state.lr_scheduler_state_dict is not None and self._lr_scheduler is not None:
                self._lr_scheduler.load_state_dict(training_state.lr_scheduler_state_dict)
                logger.debug("Restored LR scheduler state")
        except Exception as e:
            logger.warning(f"⚠️ Failed to restore training state: {e}. Starting from step 0.")
            return False

        rng = training_state.rng_states
        if self._accelerator.num_processes > 1:
            logger.debug("Skipping RNG restore in multi-process mode (only main process state was saved)")
        else:
            if rng.torch_state is not None:
                torch.random.set_rng_state(rng.torch_state)
            if rng.cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng.cuda_state)
            logger.debug("Restored RNG states")

        return True

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        # For FSDP + LoRA: Cast entire model to FP32.
        # FSDP requires uniform dtype across all parameters in wrapped modules.
        # In LoRA mode, PEFT creates LoRA params in FP32 while base model is BF16.
        # We cast the base model to FP32 to match the LoRA params.
        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.model.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = (
            self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
        )

        transformer.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        # Keep frozen models on CPU for memory efficiency
        self._vae_decoder = self._vae_decoder.to("cpu")
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to("cpu")

        if self._embeddings_processor is not None and torch.cuda.is_available():
            conn_dev = self._connector_device()
            self._place_embeddings_processor()
            if conn_dev != self._accelerator.device:
                logger.info(
                    "Text connectors on %s; DiT LoRA on %s (GOPEX_CONNECTOR_CUDA_DEVICE)",
                    conn_dev,
                    self._accelerator.device,
                )

        # noinspection PyTypeChecker
        self._transformer = self._accelerator.prepare(self._transformer)

        if self._config.latenthdr.enabled and self._exposure_head is not None:
            self._exposure_head = self._exposure_head.to(self._accelerator.device)
            if not self._config.latenthdr.train_exposure_head:
                self._exposure_head.eval()

        train_dev = self._accelerator.device
        if train_dev.type == "cuda":
            self._transformer.to(train_dev)
        else:
            logger.warning(
                "Accelerate device is %s — training will be extremely slow. "
                "Reboot to fix NVML/driver mismatch or unset CUDA_VISIBLE_DEVICES.",
                train_dev,
            )

        # Log GPU memory usage after model preparation
        if train_dev.type == "cuda":
            idx = train_dev.index if train_dev.index is not None else torch.cuda.current_device()
            vram_usage_gb = torch.cuda.memory_allocated(idx) / 1024**3
            logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB on cuda:{idx}")
        else:
            logger.debug("GPU memory usage after models preparation: n/a (device %s)", train_dev)

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._config.training_strategy.get_data_sources()

            caption_index = None
            if self._config.model.finetune_text_stack and self._config.model.text_stack_live_captions:
                from ltx_trainer.text_stack_utils import load_caption_index

                caption_index = load_caption_index(self._config.data.dataset_manifest_path)
            self._caption_index = caption_index
            self._dataset = PrecomputedDataset(
                self._config.data.preprocessed_data_root,
                data_sources=data_sources,
                caption_index=caption_index,
            )
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")
            if caption_index:
                logger.info("Live captions: %s manifest entries", f"{len(caption_index):,}")
                if self._text_encoder is not None:
                    from ltx_trainer.text_stack_utils import gemma_caption_cache_enabled

                    if gemma_caption_cache_enabled():
                        self._gemma_caption_cache_dir = Path(self._config.output_dir) / "gemma_caption_cache"
            sampling_mode = self._config.data.project_sampling_mode
            if sampling_mode != "uniform" and getattr(self._dataset, "project_groups", None):
                from ltx_trainer.project_sampling import compute_project_sample_weights, summarize_project_groups

                groups = self._dataset.project_groups
                weights = compute_project_sample_weights(groups, mode=sampling_mode)
                summary = summarize_project_groups(groups)
                logger.info(
                    "Project sampling mode=%s across %s projects (top uniform share: %s=%.1f%% → balanced %.1f%% each)",
                    sampling_mode,
                    len(groups),
                    summary[0]["project"] if summary else "?",
                    float(summary[0]["uniform_pct"]) if summary else 0.0,
                    float(summary[0]["balanced_pct"]) if summary else 0.0,
                )
                for row in summary[:8]:
                    logger.debug(
                        "  %s: %s clips (uniform %.1f%%, %s target %.1f%%/proj)",
                        row["project"],
                        row["clips"],
                        row["uniform_pct"],
                        sampling_mode,
                        row["balanced_pct"],
                    )
                if len(summary) > 8:
                    logger.debug("  … and %s more projects", len(summary) - 8)

        num_workers = self._config.data.num_dataloader_workers
        sampler = None
        shuffle = True
        if self._dataset is not None and self._config.data.project_sampling_mode != "uniform":
            from ltx_trainer.project_sampling import compute_project_sample_weights

            weights = compute_project_sample_weights(
                self._dataset.project_groups,
                mode=self._config.data.project_sampling_mode,
            )
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False

        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler."""
        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        n_trainable = sum(p.numel() for p in self._trainable_params)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        use_8bit = opt_cfg.optimizer_type == "adamw8bit"
        if use_8bit and self._trainable_params and not all(p.is_cuda for p in self._trainable_params):
            logger.warning(
                "Trainable parameters are not on CUDA; AdamW8bit requires GPU. Falling back to AdamW."
            )
            use_8bit = False

        if opt_cfg.optimizer_type == "adamw" or (opt_cfg.optimizer_type == "adamw8bit" and not use_8bit):
            optimizer = AdamW(self._trainable_params, lr=lr)
            if opt_cfg.optimizer_type == "adamw8bit":
                logger.info("Using AdamW for %s trainable parameters", f"{n_trainable:,}")
        elif use_8bit:
            # noinspection PyUnresolvedReferences
            from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

            logger.info(
                "Using AdamW8bit for %s trainable parameters (saves VRAM vs full Adam state)",
                f"{n_trainable:,}",
            )
            optimizer = AdamW8bit(self._trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        lr_scheduler = self._create_scheduler(optimizer)

        # noinspection PyTypeChecker
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(optimizer, lr_scheduler)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = dict(self._config.optimization.scheduler_params or {})
        # warmup_steps is a Gopex/diffusers-style param; PyTorch cosine/step schedulers
        # do not accept it — apply via SequentialLR when present.
        warmup_steps = int(params.pop("warmup_steps", 0) or 0)

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            cosine_steps = max(steps - warmup_steps, 1)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", max(steps // 4, 1)),
                T_mult=params.pop("T_mult", 1),
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        if scheduler is not None and warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-2,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
                milestones=[warmup_steps],
            )

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

            local_batch = self._config.optimization.batch_size
            global_batch = self._config.optimization.batch_size * self._accelerator.num_processes
            logger.info(f"Local batch size: {local_batch}, global batch size: {global_batch}")

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.acceleration.quantization:
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    # Note: Use @torch.no_grad() instead of @torch.inference_mode() to avoid FSDP inplace update errors after validation
    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _sample_videos(self, progress: TrainingProgress) -> list[Path] | None:
        """Run validation by generating videos from validation prompts."""
        use_images = self._config.validation.images is not None
        use_reference_videos = self._config.validation.reference_videos is not None
        generate_audio = self._config.validation.generate_audio
        inference_steps = self._config.validation.inference_steps

        # Zero gradients and free GPU memory to reclaim memory before validation sampling
        self._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        # Start sampling progress tracking
        sampling_ctx = progress.start_sampling(
            num_prompts=len(self._config.validation.prompts),
            num_steps=inference_steps,
        )

        # Create validation sampler with loaded models and progress tracking
        sampler = ValidationSampler(
            transformer=self._transformer,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,
            audio_decoder=self._audio_vae if generate_audio else None,
            vocoder=self._vocoder if generate_audio else None,
            sampling_context=sampling_ctx,
            embeddings_processor=self._embeddings_processor,
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        video_paths = []
        width, height, num_frames = self._config.validation.video_dims

        for prompt_idx, prompt in enumerate(self._config.validation.prompts):
            # Update progress to show current video
            sampling_ctx.start_video(prompt_idx)

            # Load conditioning image if provided
            condition_image = None
            if use_images:
                image_path = self._config.validation.images[prompt_idx]
                image = open_image_as_srgb(image_path)
                # Convert PIL image to tensor [C, H, W] in [0, 1]
                condition_image = F.to_tensor(image)

            # Load reference video if provided (for IC-LoRA)
            reference_video = None
            if use_reference_videos:
                ref_video_path = self._config.validation.reference_videos[prompt_idx]
                # read_video returns [F, C, H, W] in [0, 1]
                reference_video, _ = read_video(ref_video_path, max_frames=num_frames)

            # Get cached embeddings for this prompt if available
            cached_embeddings = (
                self._cached_validation_embeddings[prompt_idx]
                if self._cached_validation_embeddings is not None
                else None
            )

            # Create generation config
            gen_config = GenerationConfig(
                prompt=prompt,
                negative_prompt=self._config.validation.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=self._config.validation.frame_rate,
                num_inference_steps=inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed,
                condition_image=condition_image,
                reference_video=reference_video,
                reference_downscale_factor=self._config.validation.reference_downscale_factor,
                generate_audio=generate_audio,
                include_reference_in_output=self._config.validation.include_reference_in_output,
                cached_embeddings=cached_embeddings,
                stg_scale=self._config.validation.stg_scale,
                stg_blocks=self._config.validation.stg_blocks,
                stg_mode=self._config.validation.stg_mode,
            )

            # Generate sample
            video, audio = sampler.generate(
                config=gen_config,
                device=self._accelerator.device,
            )

            # Save output (image for single frame, video otherwise)
            if IS_MAIN_PROCESS:
                ext = "png" if num_frames == 1 else "mp4"
                output_path = output_dir / f"step_{self._global_step:06d}_{prompt_idx + 1}.{ext}"
                if num_frames == 1:
                    save_image(video, output_path)
                else:
                    save_video(
                        video_tensor=video,
                        output_path=output_path,
                        fps=self._config.validation.frame_rate,
                        audio=audio,
                        audio_sample_rate=self._vocoder.output_sampling_rate if audio is not None else None,
                    )
                video_paths.append(output_path)

        # Clean up progress tasks
        sampling_ctx.cleanup()

        # ValidationSampler parks the text stack on CPU for DiT VRAM; restore before training resumes.
        self._place_embeddings_processor()

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")
        self._maybe_assert_validation_quality(video_paths)
        return video_paths

    def _maybe_assert_validation_quality(self, sampled_videos_paths: list[Path] | None) -> None:
        """Abort training when validation MP4s look like color noise (env GOPEX_VALIDATION_QUALITY_GATE=1)."""
        if not IS_MAIN_PROCESS or not sampled_videos_paths:
            return
        gate = os.environ.get("GOPEX_VALIDATION_QUALITY_GATE", "").strip().lower()
        if gate not in ("1", "true", "yes"):
            return
        min_lap = float(os.environ.get("GOPEX_VALIDATION_MIN_LAP", "80"))
        from ltx_trainer.validation_quality import assert_samples_ok

        try:
            reports = assert_samples_ok(sampled_videos_paths, min_laplacian_var=min_lap)
        except RuntimeError as exc:
            logger.error("Validation quality gate failed at step %s: %s", self._global_step, exc)
            raise SystemExit(2) from exc
        for r in reports:
            logger.info(
                "Validation quality OK: %s lap=%.1f luma=%.1f",
                r.path.name,
                r.laplacian_var,
                r.mean_luma,
            )

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _text_stack_freeze_dit_enabled(self) -> bool:
        return bool(self._config.model.finetune_text_stack and self._config.model.text_stack_freeze_dit)

    def _save_checkpoint(self) -> Path | None:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP
        freeze_dit = self._text_stack_freeze_dit_enabled()

        # Prepare paths
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Get state dict (collective operation - all processes must participate)
        self._accelerator.wait_for_everyone()

        if freeze_dit:
            if not IS_MAIN_PROCESS:
                return None
            save_dir.mkdir(exist_ok=True, parents=True)
            save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32
            saved_weights_path = save_dir / f"text_stack_weights_step_{self._global_step:05d}.safetensors"
            _save_text_stack_sidecar(self._embeddings_processor, saved_weights_path, save_dtype)
            rel_path = saved_weights_path.relative_to(self._config.output_dir)
            logger.info(
                "Text stack weights for step %s saved in %s (frozen DiT — no LoRA checkpoint)",
                self._global_step,
                rel_path,
            )
            self._checkpoint_paths.append(saved_weights_path)
            self._cleanup_checkpoints()
            self._save_training_state(save_dir)
            return saved_weights_path

        full_state_dict = self._accelerator.get_state_dict(self._transformer)

        if not IS_MAIN_PROCESS:
            return None

        save_dir.mkdir(exist_ok=True, parents=True)

        # Determine save precision
        save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32

        # For LoRA: extract only adapter weights; for full: use as-is
        if is_lora:
            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            # For FSDP, pass full_state_dict since model params aren't directly accessible
            state_dict = get_peft_model_state_dict(unwrapped, state_dict=full_state_dict if is_fsdp else None)

            # Remove "base_model.model." prefix added by PEFT
            state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}

            # Convert to ComfyUI-compatible format (add "diffusion_model." prefix)
            state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}

            # Cast to configured precision
            state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in state_dict.items()}

            # Build metadata for safetensors file
            metadata = self._build_checkpoint_metadata()

            # Save to disk with metadata
            save_file(state_dict, saved_weights_path, metadata=metadata)
            if self._config.model.finetune_text_connectors:
                te_path = _text_embed_sidecar_path(saved_weights_path)
                te_sd = _text_connector_state_dict_for_save(self._embeddings_processor)
                te_sd = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in te_sd.items()}
                save_file(te_sd, te_path)
                logger.info(
                    "Text connector weights for step %s saved in %s",
                    self._global_step,
                    te_path.relative_to(self._config.output_dir),
                )
            if self._config.model.finetune_text_stack:
                ts_path = _text_stack_sidecar_path(saved_weights_path)
                _save_text_stack_sidecar(self._embeddings_processor, ts_path, save_dtype)
                logger.info(
                    "Text stack weights for step %s saved in %s",
                    self._global_step,
                    ts_path.relative_to(self._config.output_dir),
                )
        else:
            # Cast to configured precision
            full_state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in full_state_dict.items()}

            # Save to disk
            self._accelerator.save(full_state_dict, saved_weights_path)
            if self._config.model.finetune_text_connectors:
                te_path = _text_embed_sidecar_path(saved_weights_path)
                te_sd = _text_connector_state_dict_for_save(self._embeddings_processor)
                te_sd = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in te_sd.items()}
                save_file(te_sd, te_path)
                logger.info(
                    "Text connector weights for step %s saved in %s",
                    self._global_step,
                    te_path.relative_to(self._config.output_dir),
                )
            if self._config.model.finetune_text_stack:
                ts_path = _text_stack_sidecar_path(saved_weights_path)
                _save_text_stack_sidecar(self._embeddings_processor, ts_path, save_dtype)
                logger.info(
                    "Text stack weights for step %s saved in %s",
                    self._global_step,
                    ts_path.relative_to(self._config.output_dir),
                )

        rel_path = saved_weights_path.relative_to(self._config.output_dir)
        logger.info(f"💾 {prefix.capitalize()} weights for step {self._global_step} saved in {rel_path}")

        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()

        self._save_training_state(save_dir)

        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
                te = _text_embed_sidecar_path(old_checkpoint)
                if te.exists():
                    te.unlink()
                    logger.info(f"Removed old text-embed checkpoint: {te}")
                ts = _text_stack_sidecar_path(old_checkpoint)
                if ts.exists():
                    ts.unlink()
                    logger.info(f"Removed old text-stack checkpoint: {ts}")
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _save_training_state(self, save_dir: Path) -> None:
        """Save training state alongside checkpoint for resume.
        Respects checkpoints.save_training_state config:
        - "full": optimizer + scheduler + RNG + step + wandb_run_id
        - "minimal": scheduler + RNG + step + wandb_run_id
        - "off": skip entirely
        """
        if not IS_MAIN_PROCESS:
            return

        mode = self._config.checkpoints.save_training_state
        if mode == "off":
            return

        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        optimizer_state = None
        if mode == "full":
            if is_fsdp:
                logger.warning(
                    "⚠️ save_training_state='full' is not supported with FSDP. "
                    "Saving 'minimal' state (scheduler + RNG only)."
                )
            else:
                optimizer_state = self._optimizer.state_dict()

        state = TrainingState(
            global_step=self._global_step,
            config_fingerprint=ConfigFingerprint(
                optimizer_type=self._config.optimization.optimizer_type,
                scheduler_type=self._config.optimization.scheduler_type,
                training_mode=self._config.model.training_mode,
                lora_rank=self._config.lora.rank if self._config.lora is not None else None,
                finetune_text_connectors=self._config.model.finetune_text_connectors,
                finetune_text_stack=self._config.model.finetune_text_stack,
            ),
            rng_states=RngStates(
                torch_state=torch.random.get_rng_state(),
                cuda_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            ),
            lr_scheduler_state_dict=self._lr_scheduler.state_dict() if self._lr_scheduler is not None else None,
            optimizer_state_dict=optimizer_state,
            wandb_run_id=self._wandb_run.id if self._wandb_run is not None else None,
        )

        state_path = save_dir / f"training_state_step_{self._global_step:05d}.pt"
        tmp_path = state_path.with_suffix(".pt.tmp")
        try:
            torch.save(state.to_save_dict(), tmp_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        tmp_path.rename(state_path)

        file_size_gb = state_path.stat().st_size / (1024**3)
        if file_size_gb > 1.0 and not self._training_state_size_warned:
            self._training_state_size_warned = True
            logger.warning(
                f"⚠️ Training state file is {file_size_gb:.1f} GB (full mode includes optimizer state). "
                f'Set checkpoints.save_training_state="minimal" to save only scheduler/RNG/step (~few KB), '
                f'or "off" to disable entirely.'
            )

        if not self._training_state_paths or self._training_state_paths[-1] != state_path:
            self._training_state_paths.append(state_path)
        self._cleanup_training_states()

        rel_path = state_path.relative_to(self._config.output_dir)
        logger.debug(f"Training state saved to {rel_path}")

    def _cleanup_training_states(self) -> None:
        """Clean up old training state files, using the same keep_last_n as checkpoints."""
        keep_n = self._config.checkpoints.keep_last_n
        if 0 < keep_n < len(self._training_state_paths):
            to_remove = self._training_state_paths[:-keep_n]
            for old_state in to_remove:
                if old_state.exists():
                    old_state.unlink()
                    logger.debug(f"Removed old training state: {old_state}")
            self._training_state_paths = self._training_state_paths[-keep_n:]

    def _build_checkpoint_metadata(self) -> dict[str, str]:
        """Build metadata dictionary for safetensors checkpoint.
        Delegates to the training strategy to get strategy-specific metadata
        that downstream inference pipelines may need.
        Returns:
            Dictionary of string key-value pairs for safetensors metadata.
            Values are converted to strings for safetensors compatibility.
        """
        raw_metadata = self._training_strategy.get_checkpoint_metadata()
        # Convert all values to strings for safetensors compatibility
        metadata = {k: str(v) for k, v in raw_metadata.items()}
        if metadata:
            logger.info(f"Saving checkpoint metadata: {metadata}")
        return metadata

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _init_wandb(self, resume_run_id: str | None = None) -> None:
        """Initialize Weights & Biases run."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        init_kwargs: dict[str, Any] = {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "name": Path(self._config.output_dir).name,
            "tags": wandb_config.tags,
            "config": self._config.model_dump(),
        }
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "allow"
        run = wandb.init(**init_kwargs)
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def _log_validation_samples(self, sample_paths: list[Path], prompts: list[str]) -> None:
        """Log validation samples (videos or images) to Weights & Biases."""
        if not self._config.wandb.log_validation_videos or self._wandb_run is None:
            return

        # Determine if outputs are images or videos based on file extension
        is_image = sample_paths and is_still_image_path(sample_paths[0])

        if is_image:
            samples = [
                wandb.Image(str(path), caption=prompt) for path, prompt in zip(sample_paths, prompts, strict=True)
            ]
        else:
            samples = [
                wandb.Video(str(path), caption=prompt, format=path.suffix.lower().lstrip("."))
                for path, prompt in zip(sample_paths, prompts, strict=True)
            ]
        self._wandb_run.log({"validation_samples": samples}, step=self._global_step)
