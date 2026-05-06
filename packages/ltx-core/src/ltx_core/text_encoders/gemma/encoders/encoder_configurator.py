"""Gemma 4 text encoder wiring for LTX (configurator, weight key ops, meta-model buffers).

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
    - Use an LTX / embeddings-processor checkpoint trained for your Gemma 4 width and depth (feature
      extractor ``flat_dim`` follows ``Gemma4Config().text_config`` defaults from your Transformers
      version unless you extend the LTX config).
"""
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

        # Create video embeddings connector (always needed)
        video_connector = Embeddings1DConnectorConfigurator.from_config(config)

        # Create audio embeddings connector
        audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(config)

        # Create feature extractor
        feature_extractor = _create_feature_extractor(transformer_config)

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


def _gemma4_text_dims_for_feature_extractor() -> tuple[int, int]:
    """Hidden size and number of hidden-state slots (decoder layers + embedding) for the feature extractor."""
    text_cfg = Gemma4Config().text_config
    return text_cfg.hidden_size, text_cfg.num_hidden_layers + 1


def _create_feature_extractor(transformer_config: dict) -> torch.nn.Module:
    """Select and create the appropriate feature extractor based on config.
    Detection logic:
    - V1: V2 config keys absent → projection lives in transformer
    - V2: V2 config keys present with exact expected values → per-token RMS norm with dual aggregate embeds
    - Anything else: NotImplementedError (config drift)

    Gemma 4 text width and depth come from the active ``Gemma4TextConfig`` defaults in Transformers so that
    ``flat_dim`` matches a meta model built from the same library version. Your LTX checkpoint must be
    trained for those dimensions (or extend the checkpoint config to carry explicit Gemma text sizes).
    """
    embedding_dim, num_layers = _gemma4_text_dims_for_feature_extractor()
    flat_dim = embedding_dim * num_layers

    overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
    if not overlapping_keys:
        aggregate_embed = torch.nn.Linear(flat_dim, embedding_dim, bias=False)
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
    return FeatureExtractorV2(
        video_aggregate_embed=torch.nn.Linear(flat_dim, video_inner_dim, bias=True),
        embedding_dim=embedding_dim,
        audio_aggregate_embed=torch.nn.Linear(flat_dim, audio_inner_dim, bias=True),
    )


# --- Split SDOps: Gemma LLM keys vs Embeddings Processor keys ---

GEMMA_LLM_KEY_OPS = (
    SDOps("GEMMA4_LLM_KEY_OPS")
    # Map flattened / partial checkpoints onto GemmaTextEncoder.model (Gemma4ForConditionalGeneration).
    # Works for both ``language_model.layers.*`` (Gemma4TextModel) and ``language_model.model.layers.*``
    # (Gemma4ForCausalLM-style exports): a single ``language_model.`` prefix maps to ``model.model.language_model.``.
    .with_matching(prefix="language_model.")
    .with_replacement("language_model.", "model.model.language_model.")
    # Vision / fusion (Gemma 4 naming)
    .with_matching(prefix="vision_tower.")
    .with_replacement("vision_tower.", "model.vision_tower.")
    .with_matching(prefix="embed_vision.")
    .with_replacement("embed_vision.", "model.model.embed_vision.")
    .with_matching(prefix="embed_audio.")
    .with_replacement("embed_audio.", "model.model.embed_audio.")
    # Legacy Gemma 3 multimodal projector keys in some repacked checkpoints
    .with_matching(prefix="multi_modal_projector.")
    .with_replacement("multi_modal_projector.", "model.model.embed_vision.")
    # Duplicate embed_tokens to lm_head (needed for prompt enhancement via generate())
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
