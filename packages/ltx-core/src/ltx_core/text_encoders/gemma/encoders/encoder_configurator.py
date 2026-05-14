"""Gemma 4 text encoder wiring for LTX (configurator, weight key ops, meta-model buffers).

Experimental mismatch bridge (pipelines only, opt-in via env in ``ltx_pipelines``):
    When Gemma ``encode()`` flat dim (``hidden_size * num_stack``) differs from the LTX checkpoint
    ``video_aggregate_embed.in_features``, set ``ltx_checkpoint_text_flat_dim`` and
    ``ltx_experimental_flat_dim_bridge`` on the embeddings checkpoint dict and pass
    ``checkpoint_text_flat_dim`` / ``experimental_flat_dim_bridge`` into ``_create_feature_extractor``.
    Optionally set ``ltx_experimental_flat_dim_bridge_rank`` for a low-rank bottleneck instead of one huge
    dense ``nn.Linear``. Otherwise a dense random bridge is used (sub-optimal quality; impractical to train).

What changed from Gemma 3:
    - ``Gemma3ForConditionalGeneration`` / ``Gemma3Config`` → ``Gemma4ForConditionalGeneration`` /
      ``Gemma4Config`` (Transformers).
    - Vision fusion keys: ``multi_modal_projector`` → ``embed_vision`` / ``embed_audio`` under
      ``model.model``; legacy projector keys are still remapped for older repacks.
    - Language-model keys: any prefix ``language_model.`` is mapped to ``model.model.language_model.``
      (covers both ``Gemma4TextModel`` and ``Gemma4ForCausalLM``-style state dicts).
    - ``create_and_populate`` no longer touches SigLIP-style vision buffers; it refreshes text RoPE
      inverse-frequency buffers and the token embedding scale on CPU after meta init.

What you must change when upgrading checkpoints:
    - Ship Hugging Face Gemma 4 ``config.json`` (or safetensors ``config`` metadata) next to weights;
      see ``resolve_gemma_checkpoint_config`` in ``config.py``.
    - Use an LTX / embeddings-processor checkpoint trained for your Gemma 4 width and depth. The feature
      extractor ``flat_dim`` follows ``text_config`` in ``gemma_hf_config`` when present; pipelines can inject
      ``ltx_encode_stack_dims`` (from a real ``output_hidden_states`` forward) when JSON metadata disagrees with
      loaded tensors. Otherwise ``Gemma4Config()`` defaults apply.
"""
import math

import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

from ltx_core.loader import KeyValueOperationResult
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.text_encoders.gemma.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
from ltx_core.text_encoders.gemma.feature_extractor import (
    FeatureExtractorV1,
    FeatureExtractorV2,
)


class GemmaTextEncoderConfigurator(ModelConfigurator[GemmaTextEncoder]):
    @classmethod
    def from_config(cls, config: dict) -> GemmaTextEncoder:
        gemma_config = Gemma4Config.from_dict(config)
        with torch.device("meta"):
            model = Gemma4ForConditionalGeneration(gemma_config)

        return GemmaTextEncoder(model=model)


class EmbeddingsProcessorConfigurator(ModelConfigurator[EmbeddingsProcessor]):
    @classmethod
    def from_config(cls, config: dict) -> EmbeddingsProcessor:
        transformer_config = config.get("transformer", {})
        gemma_hf_config = config.get("gemma_hf_config")
        gemma_merged: dict = dict(gemma_hf_config) if isinstance(gemma_hf_config, dict) else {}
        top_stack = config.get("ltx_encode_stack_dims")
        if isinstance(top_stack, dict):
            gemma_merged["ltx_encode_stack_dims"] = top_stack

        # Create video embeddings connector (always needed)
        video_connector = Embeddings1DConnectorConfigurator.from_config(config)

        # Create audio embeddings connector
        audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(config)

        # Create feature extractor
        stack_override = config.get("ltx_encode_stack_dims")
        ck_flat_raw = config.get("ltx_checkpoint_text_flat_dim")
        ck_flat = int(ck_flat_raw) if ck_flat_raw is not None else None
        bridge_flag = bool(config.get("ltx_experimental_flat_dim_bridge"))
        bridge_rank_raw = config.get("ltx_experimental_flat_dim_bridge_rank")
        bridge_rank = int(bridge_rank_raw) if bridge_rank_raw is not None else None
        feature_extractor = _create_feature_extractor(
            transformer_config,
            gemma_hf_config=gemma_merged if gemma_merged else None,
            stack_dims_override=stack_override if isinstance(stack_override, dict) else None,
            checkpoint_text_flat_dim=ck_flat,
            experimental_flat_dim_bridge=bridge_flag,
            flat_dim_bridge_rank=bridge_rank,
        )

        return EmbeddingsProcessor(
            video_connector=video_connector,
            audio_connector=audio_connector,
            feature_extractor=feature_extractor,
        )


_V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


def _gemma_text_stack_dims(gemma_hf_config: dict | None) -> tuple[int, int]:
    """Return ``(hidden_size, stack_len)`` for Gemma text hidden-state stacking (``flat_dim = hidden_size * stack_len``).

    *stack_len* is the number of tensors in ``output_hidden_states`` (embedding + decoder layers), i.e.
    ``num_hidden_layers + 1`` when taken from ``text_config``.

    When *gemma_hf_config* is the resolved Hugging Face Gemma 4 JSON (see ``resolve_gemma_checkpoint_config``),
    dimensions normally come from ``text_config`` so ``flat_dim`` matches the documented Gemma tree.

    If the dict contains ``ltx_encode_stack_dims`` (``{"hidden_size", "num_stack"}`` from a real Gemma forward),
    those values win over ``text_config`` so the embeddings processor matches ``output_hidden_states`` layout even
    when JSON metadata disagrees with loaded tensors.

    If *gemma_hf_config* is omitted, fall back to current Transformers ``Gemma4Config()`` defaults (tests / legacy).
    """
    if gemma_hf_config:
        enc_stack = gemma_hf_config.get("ltx_encode_stack_dims")
        if isinstance(enc_stack, dict):
            hs = enc_stack.get("hidden_size")
            ns = enc_stack.get("num_stack")
            if hs is not None and ns is not None:
                return int(hs), int(ns)
        tc = gemma_hf_config.get("text_config")
        if isinstance(tc, dict):
            hs = tc.get("hidden_size")
            nl = tc.get("num_hidden_layers")
            if hs is not None and nl is not None:
                return int(hs), int(nl) + 1
    text_cfg = Gemma4Config().text_config
    return text_cfg.hidden_size, text_cfg.num_hidden_layers + 1


def _create_feature_extractor(
    transformer_config: dict,
    gemma_hf_config: dict | None = None,
    stack_dims_override: dict[str, int] | None = None,
    *,
    checkpoint_text_flat_dim: int | None = None,
    experimental_flat_dim_bridge: bool = False,
    flat_dim_bridge_rank: int | None = None,
) -> torch.nn.Module:
    """Select and create the appropriate feature extractor based on config.
    Detection logic:
    - V1: V2 config keys absent → projection lives in transformer
    - V2: V2 config keys present with exact expected values → per-token RMS norm with dual aggregate embeds
    - Anything else: NotImplementedError (config drift)

    Gemma text width and depth for ``flat_dim`` come from *stack_dims_override* (pipeline / checkpoint), then
    *gemma_hf_config* (``ltx_encode_stack_dims`` or ``text_config``), else Transformers ``Gemma4Config()`` defaults.

    When *checkpoint_text_flat_dim* is set and differs from the stacked Gemma flat dim, the V2 path can insert
    a random bridge (only if *experimental_flat_dim_bridge* is true) so pretrained aggregate weights still load.
    Use *flat_dim_bridge_rank* for a bottleneck ``Linear → Linear`` instead of one dense matrix (trainable / VRAM).
    """
    if isinstance(stack_dims_override, dict):
        hs = stack_dims_override.get("hidden_size")
        ns = stack_dims_override.get("num_stack")
        if hs is not None and ns is not None:
            embedding_dim = int(hs)
            num_layers = int(ns)
            flat_dim_from_stack = embedding_dim * num_layers
        else:
            embedding_dim, num_layers = _gemma_text_stack_dims(gemma_hf_config)
            flat_dim_from_stack = embedding_dim * num_layers
    else:
        embedding_dim, num_layers = _gemma_text_stack_dims(gemma_hf_config)
        flat_dim_from_stack = embedding_dim * num_layers

    aggregate_in = int(checkpoint_text_flat_dim) if checkpoint_text_flat_dim is not None else flat_dim_from_stack
    needs_bridge = aggregate_in != flat_dim_from_stack
    flat_dim_bridge: torch.nn.Module | None = None
    if needs_bridge:
        if not experimental_flat_dim_bridge:
            raise ValueError(
                f"LTX checkpoint text aggregate expects flat_dim={aggregate_in}, but Gemma hidden-state stack "
                f"resolves to {flat_dim_from_stack} (= {embedding_dim}×{num_layers}). "
                "Use a matched Gemma tree / LTX checkpoint, or set environment variable "
                "LTX_EXPERIMENTAL_ENCODE_FLAT_BRIDGE=1 (see ltx_pipelines PromptEncoder) for an un-trained bridge "
                "Linear (experimental; sub-optimal prompt adherence)."
            )
        # Dense bridge at ~87k×188k is tens of GiB in bf16 and impractical to train; prefer flat_dim_bridge_rank.
        if flat_dim_bridge_rank is not None:
            if flat_dim_bridge_rank < 2:
                raise ValueError("flat_dim_bridge_rank must be >= 2 when set")
            flat_dim_bridge = torch.nn.Sequential(
                torch.nn.Linear(flat_dim_from_stack, flat_dim_bridge_rank, bias=False, dtype=torch.bfloat16),
                torch.nn.Linear(flat_dim_bridge_rank, aggregate_in, bias=False, dtype=torch.bfloat16),
            )
            for layer in flat_dim_bridge:
                assert isinstance(layer, torch.nn.Linear)
                torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        else:
            # Default nn.Linear uses float32 weights (~61 GiB for a 87k×188k matrix). LTX pipelines run in bf16;
            # materializing the bridge in bf16 avoids a peak VRAM spike when moving the embeddings processor to CUDA.
            flat_dim_bridge = torch.nn.Linear(
                flat_dim_from_stack, aggregate_in, bias=False, dtype=torch.bfloat16
            )
            torch.nn.init.kaiming_uniform_(flat_dim_bridge.weight, a=math.sqrt(5))

    overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
    if not overlapping_keys:
        if needs_bridge:
            raise NotImplementedError(
                "Experimental flat_dim bridge is only implemented for V2 feature extractors (LTX 2.x two-stage)."
            )
        aggregate_embed = torch.nn.Linear(flat_dim_from_stack, embedding_dim, bias=False)
        return FeatureExtractorV1(aggregate_embed=aggregate_embed, is_av=True)

    missing_keys = _V2_EXPECTED_CONFIG.keys() - overlapping_keys
    if missing_keys:
        raise NotImplementedError("Partial V2 config — missing keys: " + ", ".join(sorted(missing_keys)))

    unexpected_value_keys = {k for k in overlapping_keys if transformer_config[k] != _V2_EXPECTED_CONFIG[k]}
    if unexpected_value_keys:
        raise NotImplementedError(
            "Unknown config: "
            + ", ".join(
                f"{k}={transformer_config[k]!r} (expected {_V2_EXPECTED_CONFIG[k]!r})" for k in unexpected_value_keys
            )
        )

    video_inner_dim = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
    audio_inner_dim = transformer_config["audio_num_attention_heads"] * transformer_config["audio_attention_head_dim"]
    if flat_dim_bridge is not None:
        return FeatureExtractorV2(
            video_aggregate_embed=torch.nn.Linear(aggregate_in, video_inner_dim, bias=True, dtype=torch.bfloat16),
            embedding_dim=embedding_dim,
            audio_aggregate_embed=torch.nn.Linear(aggregate_in, audio_inner_dim, bias=True, dtype=torch.bfloat16),
            flat_dim_bridge=flat_dim_bridge,
        )
    return FeatureExtractorV2(
        video_aggregate_embed=torch.nn.Linear(aggregate_in, video_inner_dim, bias=True),
        embedding_dim=embedding_dim,
        audio_aggregate_embed=torch.nn.Linear(aggregate_in, audio_inner_dim, bias=True),
        flat_dim_bridge=flat_dim_bridge,
    )


# --- Split SDOps: Gemma LLM keys vs Embeddings Processor keys ---

GEMMA_LLM_KEY_OPS = (
    SDOps("GEMMA4_LLM_KEY_OPS")
    # Native Hugging Face ``Gemma4ForConditionalGeneration`` shards use ``model.language_model.*`` / ``lm_head.*``.
    # ``GemmaTextEncoder`` nests that module at attribute ``model``, so loaded keys need an extra ``model.`` prefix.
    .with_matching(prefix="model.language_model.")
    .with_matching(prefix="model.vision_tower.")
    .with_matching(prefix="model.embed_vision.")
    .with_matching(prefix="model.embed_audio.")
    .with_matching(prefix="lm_head.")
    # Flattened / partial exports without the outer ``model.`` wrapper.
    .with_matching(prefix="language_model.")
    .with_matching(prefix="vision_tower.")
    .with_matching(prefix="embed_vision.")
    .with_matching(prefix="embed_audio.")
    .with_matching(prefix="multi_modal_projector.")
    # HF keys → ``GemmaTextEncoder`` state dict.
    .with_replacement("model.language_model.", "model.model.language_model.", prefix_only=True)
    .with_replacement("model.vision_tower.", "model.model.vision_tower.", prefix_only=True)
    .with_replacement("model.embed_vision.", "model.model.embed_vision.", prefix_only=True)
    .with_replacement("model.embed_audio.", "model.model.embed_audio.", prefix_only=True)
    .with_replacement("lm_head.", "model.lm_head.", prefix_only=True)
    # Flat keys → nested under ``GemmaTextEncoder.model``.
    .with_replacement("language_model.", "model.model.language_model.", prefix_only=True)
    .with_replacement("vision_tower.", "model.model.vision_tower.", prefix_only=True)
    .with_replacement("embed_vision.", "model.model.embed_vision.", prefix_only=True)
    .with_replacement("embed_audio.", "model.model.embed_audio.", prefix_only=True)
    .with_replacement("multi_modal_projector.", "model.model.embed_vision.", prefix_only=True)
    # Duplicate embed_tokens to lm_head (prompt enhancement via ``generate()``).
    .with_kv_operation(
        operation=lambda key, value: [
            KeyValueOperationResult(key, value),
            KeyValueOperationResult("model.lm_head.weight", value),
        ],
        key_prefix="model.model.language_model.embed_tokens.weight",
    )
)

EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("EMBEDDINGS_PROCESSOR_KEY_OPS")
    # 1. Map the feature extractor (V1: aggregate_embed inside feature_extractor)
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    # V2 dual aggregate embeds
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    .with_matching(prefix="text_embedding_projection.audio_aggregate_embed.")
    .with_replacement("text_embedding_projection.audio_aggregate_embed.", "feature_extractor.audio_aggregate_embed.")
    # 2. Map the connectors
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "video_connector.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_connector.")
)

VIDEO_ONLY_EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("VIDEO_ONLY_EMBEDDINGS_PROCESSOR_KEY_OPS")
    # 1. Map the feature extractor (V1: aggregate_embed inside feature_extractor)
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    # V2 video aggregate embed
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    # 2. Map the connectors
    .with_matching(prefix="model.diffusion_model.embeddings_connector.")
    .with_replacement("model.diffusion_model.embeddings_connector.", "embeddings_processor.video_connector.")
)


def create_and_populate(module: GemmaTextEncoder) -> GemmaTextEncoder:
    """Materialize buffers that are not always present in exported safetensors (meta → CPU tensors)."""
    model = module.model
    text_cfg = model.config.text_config
    l_model = model.model.language_model

    embed_scale = torch.tensor(text_cfg.hidden_size**0.5, device="cpu")
    l_model.embed_tokens.register_buffer("embed_scale", embed_scale)

    ref_rope = Gemma4TextRotaryEmbedding(text_cfg, device=torch.device("cpu"))
    rope = l_model.rotary_emb
    for layer_type in rope.layer_types:
        for suffix in ("_inv_freq", "_original_inv_freq"):
            name = f"{layer_type}{suffix}"
            ref_tensor = getattr(ref_rope, name).detach().clone()
            rope.register_buffer(name, ref_tensor, persistent=False)

    return module


GEMMA_MODEL_OPS = ModuleOps(
    name="GemmaModel",
    matcher=lambda module: hasattr(module, "model") and isinstance(module.model, Gemma4ForConditionalGeneration),
    mutator=create_and_populate,
)
