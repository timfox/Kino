"""Pipeline blocks — each block owns its model lifecycle.
Blocks build a model on each ``__call__``, use it, then free GPU memory.
This eliminates manual ``del model; cleanup_memory()`` in pipelines and
removes the need for :class:`ModelLedger`.
"""

from __future__ import annotations

import copy
import logging
import os
from collections import Counter
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Callable, TypeVar

import torch
from safetensors.torch import load_file

from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.block_streaming import DISK_CPU_SLOTS, StreamingModelBuilder
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import Noiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import SDOps
from ltx_core.loader.helpers import (
    peek_video_aggregate_embed_in_features,
    read_ltx_checkpoint_config,
    resolve_audio_vae_checkpoint_path,
    resolve_ltx_checkpoint_paths,
    resolve_vae_checkpoint_path,
)
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.loader.attention_ops import set_attention_module_op
from ltx_core.loader.fuse_loras import bf16_fuse_rule
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import BuilderProtocol, LoraPathStrengthAndSDOps, ModelBuilderProtocol
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae import (
    decode_audio as vae_decode_audio,
)
from ltx_core.model.model_protocol import LTXModelProtocol, ModelConfigurator
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.transformer.attention import (
    AttentionCallable,
    AttentionFunction,
)
from ltx_core.model.transformer.compiling import (
    CompilationConfig,
    build_compile_transformer_op,
    modify_sd_ops_for_compilation,
)
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import (
    MEMORY_EFFICIENT_DECODE,
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.quantization import QuantizationPolicy, fp8_cast_fuse_rule
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.text_encoders.gemma.config import resolve_gemma_checkpoint_config
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor, EmbeddingsProcessorOutput
from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.utils import find_matching_file
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    create_noised_state,
    generate_enhanced_prompt,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import Denoiser, ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)

_EXPERIMENTAL_ENCODE_FLAT_BRIDGE_ENV = "LTX_EXPERIMENTAL_ENCODE_FLAT_BRIDGE"
_LTX_STATE_DICT_LOADER = SafetensorsModelStateDictLoader()


def _ltx_checkpoint_paths(checkpoint_path: str) -> tuple[str, ...]:
    return resolve_ltx_checkpoint_paths(checkpoint_path)


def _ltx_checkpoint_config(checkpoint_path: str) -> dict:
    return read_ltx_checkpoint_config(checkpoint_path, _LTX_STATE_DICT_LOADER)


def _video_vae_checkpoint(checkpoint_path: str) -> str:
    """Standalone DiffVAE/ConvVAE for LTX-2.5 splits; unified 2.3 ckpt otherwise."""
    try:
        return str(resolve_vae_checkpoint_path(checkpoint_path))
    except FileNotFoundError:
        return checkpoint_path


def _audio_vae_checkpoint(checkpoint_path: str) -> str:
    """Standalone audio VAE+vocoder for LTX-2.5 splits; unified 2.3 ckpt otherwise."""
    try:
        return str(resolve_audio_vae_checkpoint_path(checkpoint_path))
    except FileNotFoundError:
        return checkpoint_path


def _ltx_builder(
    checkpoint_path: str,
    *,
    model_class_configurator: type[ModelConfigurator],
    model_sd_ops: SDOps,
    registry: Registry | None = None,
    module_ops: tuple[ModuleOps, ...] = (),
    loras: tuple = (),
) -> Builder:
    return Builder(
        model_path=_ltx_checkpoint_paths(checkpoint_path),
        model_class_configurator=model_class_configurator,
        model_sd_ops=model_sd_ops,
        module_ops=module_ops,
        loras=loras,
        registry=registry or DummyRegistry(),
    ).with_checkpoint_config(_ltx_checkpoint_config(checkpoint_path))

T = TypeVar("T")
_M = TypeVar("_M", bound=torch.nn.Module)


def _experimental_encode_flat_bridge_enabled() -> bool:
    """Opt-in random Linear(actual_flat → checkpoint flat) before video_aggregate_embed."""
    v = os.environ.get(_EXPERIMENTAL_ENCODE_FLAT_BRIDGE_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _peek_video_aggregate_in_features(checkpoint_path: str) -> int | None:
    return peek_video_aggregate_embed_in_features(checkpoint_path)


def _stack_dims_from_gemma_text_config_if_matches_flat(gemma_cfg: dict, flat_dim: int) -> dict[str, int] | None:
    """Return stack dims when HF text_config hidden×(layers+1) matches checkpoint flat_dim."""
    tc = gemma_cfg.get("text_config")
    if not isinstance(tc, dict):
        return None
    hs = tc.get("hidden_size")
    nl = tc.get("num_hidden_layers")
    if hs is None or nl is None:
        return None
    h_i, nl_i, fd_i = int(hs), int(nl), int(flat_dim)
    n_stack = nl_i + 1
    if h_i > 0 and n_stack > 0 and h_i * n_stack == fd_i:
        return {"hidden_size": h_i, "num_stack": n_stack}
    return None


def _finite_embeddings_processor_output(o: EmbeddingsProcessorOutput) -> EmbeddingsProcessorOutput:
    """Replace NaN/Inf in text embedding tensors."""
    v = torch.nan_to_num(o.video_encoding)
    a = o.audio_encoding
    if a is not None:
        a = torch.nan_to_num(a)
    return EmbeddingsProcessorOutput(video_encoding=v, audio_encoding=a, attention_mask=o.attention_mask)


def _gemma_text_stack_dims_from_hidden_states(hidden_states: tuple[torch.Tensor, ...]) -> dict[str, int] | None:
    if not hidden_states:
        return None
    dims = [int(h.shape[-1]) for h in hidden_states]
    if len(set(dims)) == 1:
        return {"hidden_size": dims[0], "num_stack": len(dims)}
    common, cnt = Counter(dims).most_common(1)[0]
    slack = max(1, len(dims) // 10)
    if cnt >= len(dims) - slack:
        return {"hidden_size": common, "num_stack": len(dims)}
    return None


def _reconcile_embedding_stack_dims(
    hidden_states: tuple[torch.Tensor, ...],
    stack_dims: dict[str, int] | None,
    flat_ck: int | None,
    *,
    allow_experimental_flat_mismatch: bool = False,
) -> dict[str, int] | None:
    if not hidden_states or flat_ck is None:
        return stack_dims
    n = len(hidden_states)
    if n <= 0:
        return stack_dims
    dims = [int(h.shape[-1]) for h in hidden_states]
    if flat_ck % n != 0:
        if allow_experimental_flat_mismatch and len(set(dims)) == 1:
            return {"hidden_size": dims[0], "num_stack": n}
        return stack_dims
    d_need = flat_ck // n
    if not all(d == d_need for d in dims):
        if stack_dims is not None and stack_dims["hidden_size"] * stack_dims["num_stack"] == flat_ck:
            return stack_dims
        if allow_experimental_flat_mismatch and len(set(dims)) == 1:
            return {"hidden_size": dims[0], "num_stack": n}
        raise ValueError(
            f"LTX checkpoint expects text feature flat_dim={flat_ck} (= {d_need} * {n} slices), but Gemma "
            f"output_hidden_states last dims are {dims}."
        )
    return {"hidden_size": d_need, "num_stack": n}


def _gemma_hidden_stack_len_from_loaded_text_encoder(text_encoder: torch.nn.Module) -> int | None:
    try:
        root = getattr(text_encoder, "model", None)
        inner = getattr(root, "model", None) if root is not None else None
        lm = getattr(inner, "language_model", None) if inner is not None else None
        layers = getattr(lm, "layers", None) if lm is not None else None
        if layers is None and lm is not None:
            inner_lm = getattr(lm, "model", None)
            layers = getattr(inner_lm, "layers", None) if inner_lm is not None else None
        if layers is None:
            return None
        return int(len(layers)) + 1
    except Exception:
        return None


def _load_text_connector_sidecar(embeddings_processor: EmbeddingsProcessor, path: Path) -> None:
    """Load finetuned video/audio connector weights saved during connector evolve."""
    sd = load_file(str(path))
    v_sd = {k[len("video_connector.") :]: v for k, v in sd.items() if k.startswith("video_connector.")}
    embeddings_processor.video_connector.load_state_dict(v_sd, strict=True)
    a_sd = {k[len("audio_connector.") :]: v for k, v in sd.items() if k.startswith("audio_connector.")}
    if a_sd:
        if embeddings_processor.audio_connector is None:
            logger.warning("Text-embed sidecar has audio_connector.* keys but no audio_connector on processor; skipped")
        else:
            embeddings_processor.audio_connector.load_state_dict(a_sd, strict=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chain_quantization(
    sd_ops: SDOps,
    module_ops: tuple[ModuleOps, ...],
    quantization: QuantizationPolicy,
) -> tuple[SDOps, tuple[ModuleOps, ...]]:
    chained_sd_ops = sd_ops
    if quantization.sd_ops is not None:
        chained_sd_ops = SDOps(
            name=f"sd_ops_chain_{sd_ops.name}+{quantization.sd_ops.name}",
            mapping=(*sd_ops.mapping, *quantization.sd_ops.mapping),
        )
    return chained_sd_ops, (*module_ops, *quantization.module_ops)


def _apply_compile_ops(
    sd_ops: SDOps,
    module_ops: tuple[ModuleOps, ...],
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    number_of_layers: int,
    compilation_config: CompilationConfig,
) -> tuple[SDOps, tuple[ModuleOps, ...], tuple[LoraPathStrengthAndSDOps, ...]]:
    """Rewrite sd_ops/module_ops/LoRAs for compiled blocks (params land under ``_orig_mod``)."""
    sd_ops = modify_sd_ops_for_compilation(sd_ops, number_of_layers)
    compile_op = build_compile_transformer_op(compilation_config)
    module_ops = (*module_ops, compile_op)
    loras = tuple(
        LoraPathStrengthAndSDOps(
            lora.path,
            lora.strength,
            modify_sd_ops_for_compilation(lora.sd_ops, number_of_layers),
        )
        for lora in loras
    )
    return sd_ops, module_ops, loras


@contextmanager
def _streaming_model(
    builder: StreamingModelBuilder,
    offload_mode: OffloadMode,
    target_device: torch.device,
    dtype: torch.dtype,
) -> Iterator:
    """Build a streaming wrapper, yield it, then tear down and free memory."""
    cpu_slots_count = DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None
    wrapped = builder.build(
        device=target_device,
        dtype=dtype,
        cpu_slots_count=cpu_slots_count,
    )
    try:
        yield wrapped
    finally:
        wrapped.teardown()
        wrapped.to("meta")
        cleanup_memory()


def _build_state(
    spec: ModalitySpec,
    tools: LatentTools,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
) -> LatentState:
    """Create a noised latent state from a modality spec and tools."""
    state = create_noised_state(
        tools=tools,
        conditionings=spec.conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=spec.noise_scale,
        initial_latent=spec.initial_latent,
    )
    if spec.frozen:
        state = replace(state, denoise_mask=torch.zeros_like(state.denoise_mask))
    return state


def _cleanup_iter(it: Iterator[torch.Tensor], model: torch.nn.Module) -> Iterator[torch.Tensor]:
    """Wrap an iterator to clean up *model* memory once it is exhausted or abandoned."""
    with gpu_model(model):
        yield from it


# ---------------------------------------------------------------------------
# DiffusionStage
# ---------------------------------------------------------------------------


class DiffusionStage:
    """Owns transformer lifecycle. Builds on each call, frees on exit.
    Replaces the manual ``model_ledger.transformer()`` / ``del transformer``
    pattern in every pipeline.
    """

    def __init__(  # noqa: PLR0913
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        transformer_builder: ModelBuilderProtocol[LTXModelProtocol] | None = None,
        model_configurator: type[ModelConfigurator] = LTXModelConfigurator,
        model_sd_ops: SDOps = LTXV_MODEL_COMFY_RENAMING_MAP,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._quantization = quantization
        self._compilation_config = compilation_config
        self._offload_mode = offload_mode
        # A quantization policy may pin its own configurator; otherwise use the one
        # provided by the caller (defaults to the audio-video LTXModelConfigurator).
        configurator = (
            quantization.model_configurator
            if quantization is not None and quantization.model_configurator is not None
            else model_configurator
        )
        if transformer_builder is not None:
            self._transformer_builder = transformer_builder
        else:
            self._transformer_builder = _ltx_builder(
                checkpoint_path,
                model_class_configurator=configurator,
                model_sd_ops=model_sd_ops,
                loras=tuple(loras),
                registry=registry,
            )

        if offload_mode != OffloadMode.NONE:
            # WeightsProvider currently only supports plain bf16 + fp8_cast LoRA fusion
            # (no companion-key emission). Quantization policies that emit
            # companion keys (e.g. ``.weight_scale``) cannot be streamed yet.
            if quantization is not None and quantization.fuse_rule is not fp8_cast_fuse_rule:
                raise ValueError(
                    "Block streaming is not supported with this quantization policy "
                    "(only bf16 and fp8_cast are currently supported)."
                )
            streaming_sd_ops: SDOps = model_sd_ops
            streaming_module_ops: tuple[ModuleOps, ...] = ()
            streaming_loras = tuple(loras)

            if compilation_config:
                number_of_layers = self._transformer_builder.model_config()["transformer"]["num_layers"]
                streaming_sd_ops, streaming_module_ops, streaming_loras = _apply_compile_ops(
                    streaming_sd_ops, streaming_module_ops, streaming_loras, number_of_layers
                )
            if quantization is not None:
                streaming_sd_ops, streaming_module_ops = _chain_quantization(
                    streaming_sd_ops, streaming_module_ops, quantization
                )
            self._streaming_builder = StreamingModelBuilder(
                model_class_configurator=configurator,
                model_path=_ltx_checkpoint_paths(checkpoint_path),
                model_sd_ops=streaming_sd_ops,
                module_ops=streaming_module_ops,
                loras=streaming_loras,
                registry=registry or DummyRegistry(),
                fuse_rule=quantization.fuse_rule if quantization is not None else bf16_fuse_rule,
                blocks_attr="transformer_blocks",
                blocks_prefix="transformer_blocks",
            ).with_checkpoint_config(_ltx_checkpoint_config(checkpoint_path))

    def with_attention(self, attention: AttentionFunction | AttentionCallable | None) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` that pins the transformer build to ``attention``.
        Functional: never mutates ``self``. The returned stage shares all other
        configuration with the original; only the underlying builders' ``module_ops``
        gain a ``set_attention_module_op(attention)`` entry so subsequent transformer
        builds use that kernel. ``attention=None`` is a no-op (returns ``self``).
        """
        if attention is None:
            return self
        op = set_attention_module_op(attention)
        new = copy.copy(self)
        new._transformer_builder = self._transformer_builder.with_module_ops(
            (*self._transformer_builder.module_ops, op),
        )
        if self._offload_mode != OffloadMode.NONE:
            new._streaming_builder = self._streaming_builder.with_module_ops(
                (*self._streaming_builder.module_ops, op),
            )
        return new

    def _build_transformer(self, *, device: torch.device | None = None, **kwargs: object) -> X0Model:
        target = device or self._device
        sd_ops = self._transformer_builder.model_sd_ops
        module_ops = self._transformer_builder.module_ops
        loras = self._transformer_builder.loras
        if self._compilation_config is not None:
            number_of_layers = self._transformer_builder.model_config()["transformer"]["num_layers"]
            sd_ops, module_ops, loras = _apply_compile_ops(
                sd_ops, module_ops, loras, number_of_layers, self._compilation_config
            )
        if self._quantization is not None:
            sd_ops, module_ops = _chain_quantization(sd_ops, module_ops, self._quantization)

        builder = self._transformer_builder.with_module_ops(module_ops).with_sd_ops(sd_ops).with_loras(loras)
        if self._quantization is not None:
            builder = builder.with_fuse_rule(self._quantization.fuse_rule)
        return X0Model(builder.build(device=target, **kwargs)).to(target).eval()

    @contextmanager
    def _streaming_transformer_ctx(self) -> Iterator[X0Model]:
        with _streaming_model(
            self._streaming_builder, self._offload_mode, self._device, self._dtype
        ) as streaming_wrapper:
            yield X0Model(streaming_wrapper).eval()

    def _transformer_ctx(self, **kwargs: object) -> AbstractContextManager:
        if self._offload_mode != OffloadMode.NONE:
            return self._streaming_transformer_ctx()
        return gpu_model(self._build_transformer(**kwargs))

    def model_context(self, **kwargs: object) -> AbstractContextManager:
        """Build the transformer, yield it, then free its memory on exit.
        Keyword arguments are forwarded to the underlying builder (e.g.
        ``video_tools`` required by ``TiledDataParallelBuilder``).
        """
        return self._transformer_ctx(**kwargs)

    def run(  # noqa: PLR0913
        self,
        transformer: object,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video: ModalitySpec | None = None,
        audio: ModalitySpec | None = None,
        stepper: DiffusionStepProtocol | None = None,
        loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
        max_batch_size: int = 1,
    ) -> tuple[LatentState | None, LatentState | None]:
        """Run denoising with a pre-built transformer.
        Same semantics as ``__call__`` but accepts a pre-built transformer so
        the model can be shared across multiple calls (e.g. tiled inference
        inside a single ``model_context()`` block). Audio supports
        ``ModalitySpec(frozen=True)`` to keep the latent unchanged throughout
        denoising while still providing cross-modal context to the transformer.
        Returns ``(video_state | None, audio_state | None)`` with cleared
        conditionings and unpatchified latents for present modalities.
        """
        if video is None and audio is None:
            raise ValueError("At least one of `video` or `audio` must be provided")

        if loop is None:
            loop = euler_denoising_loop
        if stepper is None:
            stepper = EulerDiffusionStep()

        pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)

        video_state: LatentState | None = None
        video_tools: LatentTools | None = None
        if video is not None:
            v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
            video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)
            video_state = _build_state(video, video_tools, noiser, self._dtype, self._device)

        audio_state: LatentState | None = None
        audio_tools: LatentTools | None = None
        if audio is not None:
            a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
            audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
            audio_state = _build_state(audio, audio_tools, noiser, self._dtype, self._device)

        wrapped = BatchSplitAdapter(transformer, max_batch_size=max_batch_size)  # type: ignore[arg-type]
        video_state, audio_state = loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=wrapped,
            denoiser=denoiser,
        )

        if video_state is not None and video_tools is not None:
            video_state = video_tools.clear_conditioning(video_state)
            video_state = video_tools.unpatchify(video_state)
        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)

        return video_state, audio_state

    def __call__(  # noqa: PLR0913
        self,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video: ModalitySpec | None = None,
        audio: ModalitySpec | None = None,
        stepper: DiffusionStepProtocol | None = None,
        loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
        max_batch_size: int = 1,
    ) -> tuple[LatentState | None, LatentState | None]:
        """Build transformer -> run denoising loop -> free transformer.
        Returns ``(video_state | None, audio_state | None)`` with cleared
        conditionings and unpatchified latents for present modalities.
        """
        # Build video_tools up front so it can be forwarded to the transformer
        # context (required by TiledDataParallelBuilder in multi-GPU mode).
        # `run()` rebuilds its own tools internally; the duplication is cheap.
        video_tools: LatentTools | None = None
        if video is not None:
            pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
            v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
            video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)

        mode = "streaming" if self._offload_mode != OffloadMode.NONE else "standard"
        logger.info("Building transformer (%s) from %s", mode, self._checkpoint_path)
        with self._transformer_ctx(video_tools=video_tools) as transformer:
            logger.info(
                "Running denoising loop (%d steps, %dx%d %d frames @ %.1f fps)",
                len(sigmas) - 1,
                width,
                height,
                frames,
                fps,
            )
            return self.run(
                transformer,
                denoiser,
                sigmas,
                noiser,
                width,
                height,
                frames,
                fps,
                video,
                audio,
                stepper,
                loop,
                max_batch_size,
            )


# ---------------------------------------------------------------------------
# PromptEncoder
# ---------------------------------------------------------------------------


class PromptEncoder:
    """Owns text encoder + embeddings processor lifecycle.
    Loads Gemma, encodes prompts, frees Gemma, then loads the embeddings
    processor to produce final outputs.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        text_encoder_builder: BuilderProtocol | None = None,
        text_embeds_path: str | None = None,
    ) -> None:
        self._gemma_root = gemma_root
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._offload_mode = offload_mode
        self._text_embeds_path = Path(text_embeds_path) if text_embeds_path else None

        model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
        weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]
        weight_paths_t = tuple(weight_paths)
        gemma_cfg = resolve_gemma_checkpoint_config(weight_paths_t)
        ltx_loader = SafetensorsModelStateDictLoader()
        self._embeddings_ltx_ck = read_ltx_checkpoint_config(checkpoint_path, ltx_loader)
        self._embeddings_gemma_cfg = gemma_cfg
        self._embeddings_ckpt_flat_dim = peek_video_aggregate_embed_in_features(checkpoint_path)

        if text_encoder_builder is not None:
            if offload_mode != OffloadMode.NONE:
                raise ValueError(
                    "text_encoder_builder cannot be used with offload_mode != OffloadMode.NONE "
                    "because no streaming text encoder builder is available."
                )
            self._text_encoder_builder = text_encoder_builder
            self._streaming_text_encoder_builder = None
        else:
            module_ops = module_ops_from_gemma_root(gemma_root, gemma_cfg)
            self._text_encoder_builder = Builder(
                model_path=weight_paths_t,
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GEMMA_LLM_KEY_OPS,
                module_ops=(GEMMA_MODEL_OPS, *module_ops),
                registry=registry or DummyRegistry(),
            ).with_checkpoint_config(gemma_cfg)
            self._streaming_text_encoder_builder = StreamingModelBuilder(
                model_path=weight_paths_t,
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GEMMA_LLM_KEY_OPS,
                module_ops=(GEMMA_MODEL_OPS, *module_ops),
                registry=registry or DummyRegistry(),
                blocks_attr="model.model.language_model.layers",
                blocks_prefix="model.model.language_model.layers",
            ).with_checkpoint_config(gemma_cfg)
        self._embeddings_processor_builder = _ltx_builder(
            checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=registry,
        )

    def _build_text_encoder(self) -> torch.nn.Module:
        """Build the Gemma text encoder (non-streaming path)."""
        return self._text_encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def _build_embeddings_processor(self, emb_checkpoint: dict | None = None) -> EmbeddingsProcessor:
        """Build the embeddings processor on the target device."""
        builder = self._embeddings_processor_builder
        if emb_checkpoint is not None:
            builder = builder.with_checkpoint_config(emb_checkpoint)
        return builder.build(device=self._device, dtype=self._dtype).eval()

    def _text_encoder_ctx(self) -> AbstractContextManager:
        if self._offload_mode != OffloadMode.NONE:
            return _streaming_model(self._streaming_text_encoder_builder, self._offload_mode, self._device, self._dtype)
        return gpu_model(self._build_text_encoder())

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
    ) -> list[EmbeddingsProcessorOutput]:
        """Encode *prompts* through Gemma -> embeddings processor, freeing each model after use."""
        logger.info("Building text encoder from %s", self._gemma_root)
        n_stack_model: int | None = None
        with self._text_encoder_ctx() as text_encoder:
            n_stack_model = _gemma_hidden_stack_len_from_loaded_text_encoder(text_encoder)
            if enhance_first_prompt:
                prompts = list(prompts)
                prompts[0] = generate_enhanced_prompt(
                    text_encoder, prompts[0], enhance_prompt_image, seed=enhance_prompt_seed
                )
            raw_outputs = text_encoder.encode(prompts)
            raw_outputs = [
                (tuple(h.detach().to(self._device) for h in hs), mask.detach().to(self._device))
                for hs, mask in raw_outputs
            ]
        cleanup_memory()

        hs0 = raw_outputs[0][0]
        ck_path = self._checkpoint_path
        fd = peek_video_aggregate_embed_in_features(ck_path) if ck_path else self._embeddings_ckpt_flat_dim
        if fd is None:
            fd = self._embeddings_ckpt_flat_dim

        bridge_on = _experimental_encode_flat_bridge_enabled()
        stack_dims: dict[str, int] | None = None
        if bridge_on and hs0:
            inferred_bs = _gemma_text_stack_dims_from_hidden_states(hs0)
            if inferred_bs is not None:
                stack_dims = _reconcile_embedding_stack_dims(
                    hs0, inferred_bs, fd, allow_experimental_flat_mismatch=True
                )

        if stack_dims is None:
            if fd is not None:
                stack_dims = _stack_dims_from_gemma_text_config_if_matches_flat(self._embeddings_gemma_cfg, fd)
            if (
                stack_dims is None
                and fd is not None
                and n_stack_model is not None
                and n_stack_model > 0
                and fd % n_stack_model == 0
            ):
                stack_dims = {"hidden_size": fd // n_stack_model, "num_stack": n_stack_model}
            if stack_dims is None:
                stack_dims = _gemma_text_stack_dims_from_hidden_states(hs0)
                stack_dims = _reconcile_embedding_stack_dims(
                    hs0, stack_dims, fd, allow_experimental_flat_mismatch=bridge_on
                )
                n = len(hs0)
                if fd is not None and n > 0 and fd % n == 0:
                    d_ck = fd // n
                    if all(int(t.shape[-1]) == d_ck for t in hs0):
                        stack_dims = {"hidden_size": d_ck, "num_stack": n}

        use_flat_bridge = False
        if fd is not None and stack_dims is not None:
            prod = int(stack_dims["hidden_size"]) * int(stack_dims["num_stack"])
            if prod != fd:
                if bridge_on:
                    use_flat_bridge = True
                    logger.warning(
                        "%s: Gemma encode flat_dim=%s differs from checkpoint in_features=%s; using random bridge.",
                        _EXPERIMENTAL_ENCODE_FLAT_BRIDGE_ENV,
                        prod,
                        fd,
                    )
                else:
                    raise ValueError(
                        f"LTX embeddings checkpoint expects flat_dim={fd}, but Gemma resolved to "
                        f"{stack_dims['hidden_size']}×{stack_dims['num_stack']}={prod}."
                    )
        if stack_dims is not None and hs0:
            eh, en = int(stack_dims["hidden_size"]), int(stack_dims["num_stack"])
            dims = [int(t.shape[-1]) for t in hs0]
            if len(hs0) != en or any(d != eh for d in dims):
                raise ValueError(
                    f"Gemma encode() returned {len(hs0)} hidden-state tensors (last dims {dims}); "
                    f"expected {en} tensors with last_dim={eh}."
                )

        gemma_cfg_emb = dict(self._embeddings_gemma_cfg)
        if stack_dims is not None:
            gemma_cfg_emb["ltx_encode_stack_dims"] = stack_dims
        emb_checkpoint = {**self._embeddings_ltx_ck, "gemma_hf_config": gemma_cfg_emb}
        if stack_dims is not None:
            emb_checkpoint["ltx_encode_stack_dims"] = stack_dims
        if fd is not None:
            emb_checkpoint["ltx_checkpoint_text_flat_dim"] = fd
        if use_flat_bridge:
            emb_checkpoint["ltx_experimental_flat_dim_bridge"] = True

        logger.info("Text encoder done, building embeddings processor from %s", self._checkpoint_path)
        with gpu_model(self._build_embeddings_processor(emb_checkpoint)) as embeddings_processor:
            if self._text_embeds_path is not None:
                logger.info("Loading text connector sidecar from %s", self._text_embeds_path)
                _load_text_connector_sidecar(embeddings_processor, self._text_embeds_path)
            result = [
                _finite_embeddings_processor_output(embeddings_processor.process_hidden_states(hs, mask))
                for hs, mask in raw_outputs
            ]
        logger.info("Prompt encoding complete")
        return result


# ---------------------------------------------------------------------------
# ImageConditioner
# ---------------------------------------------------------------------------


class ImageConditioner:
    """Owns video encoder lifecycle.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._encoder_builder = _ltx_builder(
            _video_vae_checkpoint(checkpoint_path),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )

    def _build_encoder(self) -> VideoEncoder:
        return self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def __call__(self, fn: Callable[[VideoEncoder], T]) -> T:
        """Build video encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(self._build_encoder()) as encoder:
            return fn(encoder)


# ---------------------------------------------------------------------------
# VideoUpsampler
# ---------------------------------------------------------------------------


class VideoUpsampler:
    """Owns video encoder + spatial upsampler lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        upsampler_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._upsampler_path = upsampler_path
        self._dtype = dtype
        self._device = device
        self._encoder_builder = _ltx_builder(
            _video_vae_checkpoint(checkpoint_path),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self._upsampler_builder = Builder(
            model_path=upsampler_path,
            model_class_configurator=LatentUpsamplerConfigurator,
            registry=registry or DummyRegistry(),
        )

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Upsample *latent* using video encoder + spatial upsampler, then free both."""
        logger.info("Building video encoder + spatial upsampler from %s", self._upsampler_path)
        with (
            gpu_model(self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()) as encoder,
            gpu_model(self._upsampler_builder.build(device=self._device, dtype=self._dtype).eval()) as upsampler,
        ):
            return upsample_video(latent=latent, video_encoder=encoder, upsampler=upsampler)


# ---------------------------------------------------------------------------
# VideoDecoder
# ---------------------------------------------------------------------------


class VideoDecoder:
    """Owns video decoder lifecycle.
    Returns an iterator that cleans up the decoder after all chunks are consumed.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        memory_efficient: bool = True,
        decoder_builder: BuilderProtocol | None = None,
    ) -> None:
        self._checkpoint_path = _video_vae_checkpoint(checkpoint_path)
        self._dtype = dtype
        self._device = device
        if decoder_builder is not None:
            self._decoder_builder = decoder_builder
        else:
            self._decoder_builder = _ltx_builder(
                self._checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=registry,
                module_ops=(MEMORY_EFFICIENT_DECODE,) if memory_efficient else (),
            )

    def __call__(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode *latent* to pixel-space video chunks. Decoder freed after exhaustion."""
        logger.info("Building video decoder from %s", self._checkpoint_path)
        decoder = self._decoder_builder.build(device=self._device, dtype=self._dtype).eval()
        return _cleanup_iter(decoder.decode_video(latent, tiling_config, generator), decoder)


# ---------------------------------------------------------------------------
# AudioDecoder
# ---------------------------------------------------------------------------


class AudioDecoder:
    """Owns audio decoder + vocoder lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._checkpoint_path = _audio_vae_checkpoint(checkpoint_path)
        self._dtype = dtype
        self._device = device
        self._decoder_builder = _ltx_builder(
            self._checkpoint_path,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self._vocoder_builder = _ltx_builder(
            self._checkpoint_path,
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )

    def __call__(self, latent: torch.Tensor) -> Audio:
        """Decode audio *latent* through VAE decoder + vocoder, then free both."""
        logger.info("Building audio decoder + vocoder from %s", self._checkpoint_path)
        with (
            gpu_model(self._decoder_builder.build(device=self._device, dtype=self._dtype).eval()) as decoder,
            gpu_model(self._vocoder_builder.build(device=self._device, dtype=self._dtype).eval()) as vocoder,
        ):
            return vae_decode_audio(latent, decoder, vocoder)


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------


class AudioConditioner:
    """Owns audio encoder lifecycle.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
    Mirrors :class:`ImageConditioner` for the audio modality.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._encoder_builder = _ltx_builder(
            _audio_vae_checkpoint(checkpoint_path),
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )

    def __call__(self, fn: Callable[[torch.nn.Module], T]) -> T:
        """Build audio encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()) as encoder:
            return fn(encoder)
