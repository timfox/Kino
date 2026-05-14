"""Builder that constructs a BlockStreamingWrapper from safetensors checkpoints."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Generic

import torch
from torch import nn

from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader, LoraSource
from ltx_core.block_streaming.pool import BlockLayout, WeightPool
from ltx_core.block_streaming.provider import WeightsProvider
from ltx_core.block_streaming.source import DiskWeightSource, PinnedWeightSource, WeightSource
from ltx_core.block_streaming.utils import (
    build_pool_layouts,
    layouts_homogeneous,
    merge_block_layouts_with_checkpoint,
    resolve_attr,
)
from ltx_core.block_streaming.wrapper import BlockStreamingWrapper
from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.helpers import create_meta_model, load_state_dict, read_model_config
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDictLoader,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

logger = logging.getLogger(__name__)

DISK_CPU_SLOTS = 2
_DEFAULT_GPU_SLOTS = 2

# Hugging Face / sharded checkpoints: ``<base>-00001-of-00004.safetensors`` (index and total digit widths vary).
_HF_SHARD_FILENAME_RE = re.compile(r"^(.+)-(\d+)-of-(\d+)\.safetensors$", re.IGNORECASE)


def expand_ltx_checkpoint_safetensors_paths(checkpoint_path: str) -> tuple[str, ...]:
    """Resolve one or more ``.safetensors`` paths for builders and streaming estimates.

    If *checkpoint_path* names a single shard of a sharded export (``*-NNN-of-MMM.safetensors``),
    return **all** sibling shards in the same directory that share the same *base* and *MMM*,
    sorted by filename. Otherwise return a one-tuple of the resolved path.

    Loading only the first shard silently omits most weights; this keeps ``load_state_dict``,
    streaming scans, and ``blocks_prefix`` inference consistent with a full checkpoint.
    """
    p = Path(checkpoint_path).expanduser().resolve()
    if not p.is_file():
        return (str(p),)
    m = _HF_SHARD_FILENAME_RE.match(p.name)
    if not m:
        return (str(p),)
    base, _shard_idx, n_total_s = m.group(1), m.group(2), m.group(3)
    try:
        n_total = int(n_total_s)
    except ValueError:
        return (str(p),)
    pattern = f"{base}-*-of-{n_total_s}.safetensors"
    hits = sorted(p.parent.glob(pattern), key=lambda x: x.name)
    if len(hits) == n_total:
        out = tuple(str(x) for x in hits)
        if len(out) > 1:
            logger.info(
                "Expanded sharded checkpoint: %d safetensors (pattern %r under %s)",
                len(out),
                pattern,
                p.parent,
            )
        return out
    logger.warning(
        "Sharded checkpoint %s: expected %d files matching %r in %s, found %d — using this shard only "
        "(place all shards in the same folder or pass a merged .safetensors).",
        p.name,
        n_total,
        pattern,
        p.parent,
        len(hits),
    )
    return (str(p),)


def infer_ltx_velocity_transformer_blocks_prefix(
    model_path: str | tuple[str, ...],
    model_sd_ops: SDOps | None,
) -> str:
    """Pick ``blocks_prefix`` for LTX velocity streaming from safetensors key names.

    After ``LTXV_MODEL_COMFY_RENAMING_MAP`` strips ``model.diffusion_model.``, shards
    may use either ``velocity_model.transformer_blocks.N.*`` or ``transformer_blocks.N.*``.
    Using the wrong prefix sends **all** weights to the non-block GPU path (~90+ GiB)
    instead of streaming blocks from CPU.
    """
    import safetensors

    paths = [model_path] if isinstance(model_path, str) else list(model_path)
    has_vm_tb = False
    has_tb = False
    for sp in paths:
        try:
            with safetensors.safe_open(sp, framework="pt") as f:
                for raw in f.keys():
                    mk = model_sd_ops.apply_to_key(raw) if model_sd_ops is not None else raw
                    if mk is None:
                        continue
                    if mk.startswith("velocity_model.transformer_blocks."):
                        has_vm_tb = True
                    elif mk.startswith("transformer_blocks."):
                        has_tb = True
        except (OSError, ValueError, RuntimeError):
            continue
    if has_vm_tb:
        return "velocity_model.transformer_blocks"
    if has_tb:
        return "transformer_blocks"
    raise ValueError(
        "Could not find `velocity_model.transformer_blocks.*` or `transformer_blocks.*` keys in "
        f"{paths[0]!s} after sd_ops; refusing to guess blocks_prefix."
    )


def estimate_ltx_velocity_streaming_partition_bytes(
    model_path: str | tuple[str, ...],
    model_sd_ops: SDOps | None,
    *,
    bytes_per_float_param: int = 2,
) -> dict[str, float | int | str]:
    """Estimate non-block vs transformer-block weight bytes using safetensors shapes only.

    Mirrors the partition rules in :meth:`StreamingModelBuilder._build_pinned_source` (keys after
    ``model_sd_ops``). Uses ``bytes_per_float_param`` (default **2** for bf16/fp16-like checkpoints)
    to convert element counts to bytes — close enough for VRAM triage.

    If ``non_block_gib`` is tens of GiB while ``block_gib`` is small, ``blocks_prefix`` likely does not
    match your shards and streaming would load almost everything onto GPU (OOM during denoise).
    """
    import math

    import safetensors

    blocks_prefix = infer_ltx_velocity_transformer_blocks_prefix(model_path, model_sd_ops)
    prefix_dot = blocks_prefix + "."
    paths = [model_path] if isinstance(model_path, str) else list(model_path)
    non_bytes = 0
    block_bytes = 0
    n_non = 0
    n_bl = 0
    seen_block_idx: set[int] = set()

    for sp in paths:
        try:
            with safetensors.safe_open(sp, framework="pt") as f:
                for raw in f.keys():
                    mk = model_sd_ops.apply_to_key(raw) if model_sd_ops is not None else raw
                    if mk is None:
                        continue
                    shp = f.get_slice(raw).get_shape()
                    nelem = int(math.prod(shp)) if shp else 0
                    nbytes = nelem * int(bytes_per_float_param)
                    if mk.startswith(prefix_dot):
                        rest = mk[len(prefix_dot) :]
                        idx_str, _, _ = rest.partition(".")
                        try:
                            bi = int(idx_str)
                        except ValueError:
                            non_bytes += nbytes
                            n_non += 1
                            continue
                        seen_block_idx.add(bi)
                        block_bytes += nbytes
                        n_bl += 1
                    else:
                        non_bytes += nbytes
                        n_non += 1
        except (OSError, ValueError, RuntimeError):
            continue

    gib = 1024.0**3
    return {
        "blocks_prefix": blocks_prefix,
        "non_block_gib": float(non_bytes / gib),
        "block_gib": float(block_bytes / gib),
        "n_non_block_tensors": int(n_non),
        "n_block_tensors": int(n_bl),
        "n_distinct_block_indices": int(len(seen_block_idx)),
    }


@dataclass(frozen=True)
class StreamingModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType]):
    """Immutable builder for :class:`BlockStreamingWrapper`.
    Reads block weights from safetensors on demand.  ``cpu_slots`` and
    ``gpu_slots`` control the memory/speed trade-off (see :meth:`build`).
    Args:
        model_class_configurator: Creates the model from a config dict.
        model_path: One or more ``.safetensors`` checkpoint paths.
        model_sd_ops: Key remapping applied to safetensors keys.
        module_ops: Module-level mutations for the meta model.
        loras: LoRA adapters fused into weights at load time.
        model_loader: Strategy for reading checkpoint metadata.
        registry: Shared cache for loaded state dicts.
        blocks_attr: Dotted path to the ``nn.ModuleList`` (e.g.
            ``"velocity_model.transformer_blocks"``).
        blocks_prefix: State-dict key prefix for block weights, e.g.
            ``"velocity_model.transformer_blocks"`` or ``"transformer_blocks"``.
            Callers should match this to checkpoint keys after ``model_sd_ops``.
        state_dict_prefix: Key prefix for non-block weights
            (e.g. ``"velocity_model."``).
        model_wrapper: Optional callable wrapping the model
            (e.g. ``X0Model``).
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)

    # Streaming-specific
    blocks_attr: str = ""
    blocks_prefix: str = ""
    state_dict_prefix: str = ""
    model_wrapper: Callable[[ModelType], nn.Module] | None = None
    checkpoint_config: dict | None = None

    def with_sd_ops(self, sd_ops: SDOps | None) -> StreamingModelBuilder:
        return replace(self, model_sd_ops=sd_ops)

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> StreamingModelBuilder:
        return replace(self, module_ops=module_ops)

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> StreamingModelBuilder:
        return replace(self, loras=loras)

    def with_checkpoint_config(self, checkpoint_config: dict | None) -> StreamingModelBuilder:
        return replace(self, checkpoint_config=checkpoint_config)

    def model_config(self) -> dict:
        """Read model configuration from the checkpoint metadata."""
        if self.checkpoint_config is not None:
            return self.checkpoint_config
        return read_model_config(self.model_path, self.model_loader)

    @staticmethod
    def _non_block_load_key(key_prefix: str, key: str) -> str:
        """Map a checkpoint key to ``load_state_dict`` keys without duplicating *key_prefix*."""
        if not key_prefix or key.startswith(key_prefix):
            return key
        return key_prefix + key

    def _validate_block_indices(
        self,
        *,
        num_blocks: int,
        seen_indices: set[int],
        sample_keys: list[str],
        source: str,
    ) -> None:
        """Fail fast if streaming would load the full transformer onto the GPU by mistake."""
        if num_blocks <= 0:
            return
        if not seen_indices:
            raise ValueError(
                f"Layer streaming ({source}): no keys matched blocks_prefix={self.blocks_prefix!r}; "
                "checkpoint weights would all be loaded as non-block GPU tensors (~tens of GiB, then OOM). "
                f"Sample model keys (after sd_ops): {sample_keys[:25]!r}"
            )
        expected = set(range(num_blocks))
        if seen_indices != expected:
            raise ValueError(
                f"Layer streaming ({source}): blocks_prefix={self.blocks_prefix!r} matched indices "
                f"{sorted(seen_indices)[:8]}… but meta model has {num_blocks} blocks "
                f"(missing {sorted(expected - seen_indices)[:10]}, extra {sorted(seen_indices - expected)[:10]}). "
                f"Sample keys: {sample_keys[:20]!r}"
            )

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        """Create a model on the meta device and apply module operations."""
        return create_meta_model(self.model_class_configurator, config, module_ops)

    def build(
        self,
        target_device: torch.device,
        dtype: torch.dtype,
        cpu_slots_count: int | None = None,
        gpu_slots_count: int | None = None,
        **_kwargs: object,
    ) -> BlockStreamingWrapper:
        """Build and return a ready-to-use :class:`BlockStreamingWrapper`.
        Args:
            target_device: GPU device for compute.
            dtype: Weight dtype (e.g. ``torch.bfloat16``).
            cpu_slots_count: Number of pinned CPU buffer slots.
                ``None`` = RAM streaming (all blocks pre-loaded with LoRA fusion).
            gpu_slots_count: Number of GPU buffer slots.
                ``None`` = ``_DEFAULT_GPU_SLOTS`` (2).
        """
        if not self.blocks_prefix:
            raise ValueError("blocks_prefix must be non-empty for streaming")

        # 1. Create meta model (no weights allocated).
        config = (
            self.checkpoint_config
            if self.checkpoint_config is not None
            else read_model_config(self.model_path, self.model_loader)
        )
        meta_model: nn.Module = create_meta_model(self.model_class_configurator, config, self.module_ops)
        if self.model_wrapper is not None:
            meta_model = self.model_wrapper(meta_model)
        meta_model.eval()

        blocks = resolve_attr(meta_model, self.blocks_attr)
        meta_block_layouts = build_pool_layouts(blocks, dtype)
        num_blocks = len(blocks)

        # 2. Determine slot counts.
        cpu_slots_count = cpu_slots_count if cpu_slots_count is not None else num_blocks
        gpu_slots_count = gpu_slots_count if gpu_slots_count is not None else _DEFAULT_GPU_SLOTS

        # 3. Build source and load non-block weights.
        if cpu_slots_count >= num_blocks:
            source, lora_sources, pool_layouts = self._build_pinned_source(
                meta_model, target_device, dtype, cpu_slots_count, meta_block_layouts, num_blocks
            )
        else:
            source, lora_sources = self._build_disk_source(
                meta_model, meta_block_layouts, target_device, dtype, cpu_slots_count, num_blocks
            )
            pool_layouts = meta_block_layouts

        homogeneous = layouts_homogeneous(pool_layouts)

        # 4. Create provider and wrapper.
        copy_stream = torch.cuda.Stream(device=target_device)
        gpu_pool = WeightPool(
            gpu_slots_count,
            target_device,
            reuse_barrier=lambda event: copy_stream.wait_event(event),
            layout=pool_layouts[0] if homogeneous else None,
            per_block_layouts=None if homogeneous else pool_layouts,
        )
        provider = WeightsProvider(gpu_pool, copy_stream, target_device, source, lora_sources, self.blocks_prefix)
        return BlockStreamingWrapper(
            model=meta_model,
            blocks=blocks,
            provider=provider,
            target_device=target_device,
        )

    def _build_pinned_source(
        self,
        meta_model: nn.Module,
        target_device: torch.device,
        dtype: torch.dtype,
        cpu_slots_count: int,
        meta_block_layouts: list[BlockLayout],
        num_blocks: int,
    ) -> tuple[WeightSource, list[LoraSource], list[BlockLayout]]:
        """Pre-load all blocks into pinned CPU buffers with LoRA fusion."""
        model_sd = load_state_dict(
            self.model_path, self.model_loader, self.registry, torch.device("cpu"), self.model_sd_ops
        )

        if self.loras:
            lora_sds = [
                load_state_dict([lora.path], self.model_loader, self.registry, torch.device("cpu"), lora.sd_ops)
                for lora in self.loras
            ]
            lora_sd_and_strengths = [
                LoraStateDictWithStrength(sd, lora.strength) for sd, lora in zip(lora_sds, self.loras, strict=True)
            ]
            model_sd = apply_loras(
                model_sd=model_sd,
                lora_sd_and_strengths=lora_sd_and_strengths,
                dtype=dtype,
                destination_sd=model_sd if isinstance(self.registry, DummyRegistry) else None,
            )

        # Partition: non-block weights go to GPU, block weights go directly
        # to pinned buffers.  This avoids holding the full state dict and
        # pinned copies simultaneously.
        non_block_sd: dict[str, torch.Tensor] = {}
        block_tensors: dict[int, dict[str, torch.Tensor]] = {}
        prefix_dot = self.blocks_prefix + "."

        for key, tensor in model_sd.sd.items():
            if key.startswith(prefix_dot):
                rest = key[len(prefix_dot) :]
                idx_str, _, param_name = rest.partition(".")
                try:
                    block_idx = int(idx_str)
                except ValueError:
                    nk = self._non_block_load_key(self.state_dict_prefix, key)
                    non_block_sd[nk] = tensor.to(device=target_device, dtype=dtype)
                    continue
                block_tensors.setdefault(block_idx, {})[param_name] = tensor
            else:
                nk = self._non_block_load_key(self.state_dict_prefix, key)
                non_block_sd[nk] = tensor.to(device=target_device, dtype=dtype)

        sample_keys = sorted(model_sd.sd.keys())
        self._validate_block_indices(
            num_blocks=num_blocks,
            seen_indices=set(block_tensors.keys()),
            sample_keys=sample_keys,
            source="pinned",
        )
        n_block_bytes = sum(t.numel() * t.element_size() for b in block_tensors.values() for t in b.values())
        n_non_block_bytes = sum(t.numel() * t.element_size() for t in non_block_sd.values())
        logger.info(
            "Layer streaming partition (pinned): blocks_prefix=%r, block_tensors=%d, "
            "pinned_block_bytes=%.2e, non_block_gpu_bytes=%.2e",
            self.blocks_prefix,
            len(block_tensors),
            float(n_block_bytes),
            float(n_non_block_bytes),
        )

        pool_layouts = merge_block_layouts_with_checkpoint(meta_block_layouts, block_tensors, dtype)

        meta_model.load_state_dict(non_block_sd, strict=False, assign=True)
        del model_sd, non_block_sd

        # Pin block weights one block at a time, freeing the source tensors as we go.
        pinned: dict[int, dict[str, torch.Tensor]] = {}
        for idx in range(cpu_slots_count):
            src = block_tensors.pop(idx)
            pinned[idx] = {name: tensor.to(dtype=dtype).pin_memory() for name, tensor in src.items()}

        return PinnedWeightSource(pinned), [], pool_layouts

    def _build_disk_source(
        self,
        meta_model: nn.Module,
        meta_block_layouts: list[BlockLayout],
        target_device: torch.device,
        dtype: torch.dtype,
        cpu_slots_count: int,
        num_blocks: int,
    ) -> tuple[WeightSource, list[LoraSource]]:
        """Create a DiskWeightSource backed by a DiskBlockReader for lazy loading."""
        lora_sources = [LoraSource(lora.path, lora.sd_ops, lora.strength) for lora in self.loras]
        checkpoint_paths = list(self.model_path) if isinstance(self.model_path, tuple) else [self.model_path]
        reader = DiskTensorReader(checkpoint_paths)

        block_key_map: dict[int, list[tuple[str, str]]] = {}
        non_block_keys: list[tuple[str, str]] = []

        for sft_key in reader.keys():  # noqa: SIM118
            model_key = self.model_sd_ops.apply_to_key(sft_key) if self.model_sd_ops else sft_key
            if model_key is None:
                continue
            if model_key.startswith(self.blocks_prefix + "."):
                rest = model_key[len(self.blocks_prefix) + 1 :]
                idx_str, _, param_name = rest.partition(".")
                try:
                    block_idx = int(idx_str)
                except ValueError:
                    non_block_keys.append((sft_key, model_key))
                    continue
                block_key_map.setdefault(block_idx, []).append((sft_key, param_name))
            else:
                non_block_keys.append((sft_key, model_key))

        sample_model_keys: list[str] = []
        for sft_key in reader.keys():
            mk = self.model_sd_ops.apply_to_key(sft_key) if self.model_sd_ops else sft_key
            if mk is not None:
                sample_model_keys.append(mk)
            if len(sample_model_keys) >= 40:
                break
        self._validate_block_indices(
            num_blocks=num_blocks,
            seen_indices=set(block_key_map.keys()),
            sample_keys=sorted(set(sample_model_keys)),
            source="disk",
        )

        self._load_non_block_weights(
            reader,
            non_block_keys,
            meta_model,
            target_device,
            dtype,
            sd_ops=self.model_sd_ops,
            key_prefix=self.state_dict_prefix,
            lora_sources=lora_sources,
            matmul_device=target_device,
        )

        homogeneous_disk = layouts_homogeneous(meta_block_layouts)
        cpu_pool = WeightPool(
            cpu_slots_count,
            torch.device("cpu"),
            reuse_barrier=lambda event: event.synchronize(),
            pin_memory=True,
            layout=meta_block_layouts[0] if homogeneous_disk else None,
            per_block_layouts=None if homogeneous_disk else meta_block_layouts,
        )
        block_reader = DiskBlockReader(reader=reader, block_key_map=block_key_map, dtype=dtype)
        source = DiskWeightSource(cpu_pool, block_reader)
        return source, lora_sources

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse_lora_delta(
        model_key: str,
        tensor: torch.Tensor,
        lora_sources: list[LoraSource],
        matmul_device: torch.device | None = None,
    ) -> torch.Tensor:
        """Add all matching LoRA deltas to *tensor* in-place."""
        if not lora_sources or not model_key.endswith(".weight"):
            return tensor
        prefix = model_key[: -len(".weight")]
        device = tensor.device if tensor.device.type == "cuda" else matmul_device
        for source in lora_sources:
            delta = source.get_delta(prefix, device=device)
            if delta is not None:
                tensor = tensor.add_(delta.to(device=tensor.device, dtype=tensor.dtype))
        return tensor

    @staticmethod
    @torch.inference_mode()
    def _load_non_block_weights(
        reader: DiskTensorReader,
        non_block_keys: list[tuple[str, str]],
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        sd_ops: SDOps | None = None,
        key_prefix: str = "",
        lora_sources: list[LoraSource] | None = None,
        matmul_device: torch.device | None = None,
    ) -> None:
        """Load non-block weights into *model* on *device*."""
        state_dict: dict[str, torch.Tensor] = {}
        sources = lora_sources or []
        for sft_key, model_key in non_block_keys:
            tensor = reader.get_tensor(sft_key).to(device=device, dtype=dtype)
            tensor = StreamingModelBuilder._fuse_lora_delta(model_key, tensor, sources, matmul_device)
            if sd_ops is not None:
                for kv in sd_ops.apply_to_key_value(model_key, tensor):
                    state_dict[StreamingModelBuilder._non_block_load_key(key_prefix, kv.new_key)] = kv.new_value
                continue
            state_dict[StreamingModelBuilder._non_block_load_key(key_prefix, model_key)] = tensor
        model.load_state_dict(state_dict, strict=False, assign=True)
