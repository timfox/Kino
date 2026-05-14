import logging
import math
from dataclasses import dataclass, field, replace
from typing import Generic

import torch
from torch import nn

from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.helpers import create_meta_model, load_state_dict, read_model_config
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

logger: logging.Logger = logging.getLogger(__name__)


def _check_uninitialized(model: nn.Module) -> list[str]:
    """Return names of any parameters/buffers still on meta device."""
    names = []
    for name, param in model.named_parameters():
        if str(param.device) == "meta":
            names.append(name)
    for name, buf in model.named_buffers():
        if str(buf.device) == "meta":
            names.append(name)
    return names


def _materialize_meta_tensors(model: nn.Module, device: torch.device, dtype: torch.dtype | None) -> None:
    """Allocate storage for parameters/buffers left on ``meta`` after ``load_state_dict(..., assign=True)``.

    Checkpoints never contain keys for modules constructed only at runtime (for example the experimental
    ``flat_dim_bridge`` Linear in :class:`EmbeddingsProcessor`). Those stay meta unless we materialize them
    here; ``Module.to(device)`` cannot move meta tensors on recent PyTorch builds.
    """
    for full_name, param in list(model.named_parameters()):
        if not param.is_meta:
            continue
        parent_name, _, leaf = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        elem_dtype = dtype if dtype is not None and param.is_floating_point() else param.dtype
        new_tensor = torch.empty(tuple(param.shape), dtype=elem_dtype, device=device)
        if leaf == "weight" and isinstance(parent, nn.Linear):
            torch.nn.init.kaiming_uniform_(new_tensor, a=math.sqrt(5))
        elif leaf == "bias" and isinstance(parent, nn.Linear) and new_tensor.numel() > 0:
            fan_in = int(parent.in_features)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            torch.nn.init.uniform_(new_tensor, -bound, bound)
        else:
            torch.nn.init.zeros_(new_tensor)
        setattr(parent, leaf, nn.Parameter(new_tensor, requires_grad=param.requires_grad))

    for full_name, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        parent_name, _, leaf = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        new_buf = torch.empty(tuple(buf.shape), dtype=buf.dtype, device=device)
        if new_buf.dtype.is_floating_point:
            torch.nn.init.zeros_(new_buf)
        parent.register_buffer(leaf, new_buf)


def _load_model_weights(
    meta_model: nn.Module,
    model_path: str | tuple[str, ...],
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    loader: StateDictLoader,
    registry: Registry,
    device: torch.device,
    dtype: torch.dtype | None,
    model_sd_ops: SDOps | None = None,
    lora_load_device: torch.device | None = None,
) -> None:
    """Load base weights and fuse LoRAs into *meta_model* in-place."""
    if lora_load_device is None:
        lora_load_device = device

    # Staging shards on CPU while the target is CUDA avoids holding an entire ``sd`` dict on GPU during
    # safetensors iteration (peak VRAM can otherwise overlap prompt-encoder memory and OOM smaller GPUs).
    load_device = torch.device("cpu") if device.type == "cuda" else device
    model_sd = load_state_dict(model_path, loader, registry, load_device, model_sd_ops)

    lora_strengths = [lora.strength for lora in loras]
    if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
        sd = model_sd.sd
        if dtype is not None:
            # In-place dtype cast avoids a temporary second full copy of ``sd`` on device.
            for key in list(sd.keys()):
                sd[key] = sd[key].to(dtype=dtype)
        meta_model.load_state_dict(sd, strict=False, assign=True)
        return

    lora_state_dicts = [load_state_dict([lora.path], loader, registry, lora_load_device, lora.sd_ops) for lora in loras]
    lora_sd_and_strengths = [
        LoraStateDictWithStrength(sd, strength) for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
    ]
    final_sd = apply_loras(
        model_sd=model_sd,
        lora_sd_and_strengths=lora_sd_and_strengths,
        dtype=dtype,
        destination_sd=model_sd if isinstance(registry, DummyRegistry) else None,
    )
    meta_model.load_state_dict(final_sd.sd, strict=False, assign=True)


@dataclass(frozen=True)
class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU.
    Attributes:
        model_class_configurator: Class responsible for constructing the model from a config dict.
        model_path: Path (or tuple of shard paths) to the model's `.safetensors` checkpoint(s).
        model_sd_ops: Optional state-dict operations applied when loading the model weights.
        module_ops: Sequence of module-level mutations applied to the meta model before weight loading.
        loras: Sequence of LoRA adapters (path, strength, optional sd_ops) to fuse into the model.
        model_loader: Strategy for loading state dicts from disk. Defaults to
            :class:`SafetensorsModelStateDictLoader`.
        registry: Cache for already-loaded state dicts. Defaults to :class:`DummyRegistry` (no caching).
        lora_load_device: Device used when loading LoRA weight tensors from disk. Defaults to
            ``torch.device("cpu")``, which keeps LoRA weights in CPU memory and transfers them to
            the target GPU sequentially during fusion, reducing peak GPU memory usage compared to
            loading all LoRA weights directly onto the GPU at once.
        checkpoint_config: When set, used as the model JSON config instead of reading metadata from
            ``model_path``. Used by the Gemma text encoder to supply ``config.json`` when safetensors
            metadata is empty.
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)
    lora_load_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    checkpoint_config: dict | None = None

    def lora(self, lora_path: str, strength: float = 1.0, sd_ops: SDOps | None = None) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def with_sd_ops(self, sd_ops: SDOps | None) -> "SingleGPUModelBuilder":
        return replace(self, model_sd_ops=sd_ops)

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> "SingleGPUModelBuilder":
        return replace(self, module_ops=module_ops)

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "SingleGPUModelBuilder":
        return replace(self, loras=loras)

    def with_registry(self, registry: Registry) -> "SingleGPUModelBuilder":
        return replace(self, registry=registry)

    def with_lora_load_device(self, device: torch.device) -> "SingleGPUModelBuilder":
        return replace(self, lora_load_device=device)

    def with_checkpoint_config(self, checkpoint_config: dict | None) -> "SingleGPUModelBuilder":
        return replace(self, checkpoint_config=checkpoint_config)

    def model_config(self) -> dict:
        if self.checkpoint_config is not None:
            return self.checkpoint_config
        return read_model_config(self.model_path, self.model_loader)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        return create_meta_model(self.model_class_configurator, config, module_ops)

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    ) -> StateDict:
        return load_state_dict(paths, self.model_loader, registry, device, sd_ops)

    def _return_model(self, meta_model: ModelType, device: torch.device, dtype: torch.dtype | None) -> ModelType:
        uninitialized = _check_uninitialized(meta_model)
        if uninitialized:
            logger.debug("Materializing meta tensors left after checkpoint load: %s", uninitialized)
            _materialize_meta_tensors(meta_model, device, dtype)
            still = _check_uninitialized(meta_model)
            if still:
                logger.warning("Parameters or buffers still on meta after materialization: %s", still)
        return meta_model.to(device)

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> ModelType:
        device = torch.device("cuda") if device is None else device
        config = self.model_config()
        meta_model = self.meta_model(config, self.module_ops)

        _load_model_weights(
            meta_model=meta_model,
            model_path=self.model_path,
            loras=self.loras,
            loader=self.model_loader,
            registry=self.registry,
            device=device,
            dtype=dtype,
            model_sd_ops=self.model_sd_ops,
            lora_load_device=self.lora_load_device,
        )
        return self._return_model(meta_model, device, dtype)
