import contextlib
import functools
import json
import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.text_encoders.gemma.config import effective_gemma_encode_max_length, resolve_gemma_checkpoint_config
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.utils import find_matching_file


def _int_token_id_for_generate(value: Any) -> int:
    """Coerce a Hugging Face token id to ``int`` for ``generate`` kwargs (never pass meta tensors through)."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta":
            msg = (
                "Token id tensor is on the meta device; use tokenizer integer ids when calling generate(). "
                "This usually means generation_config pulled meta tensors from a lazily initialized model."
            )
            raise ValueError(msg)
        if value.numel() != 1:
            msg = f"Expected a scalar token-id tensor, got shape {tuple(value.shape)}"
            raise ValueError(msg)
        return int(value.detach().cpu().item())
    return int(value)


def _generation_special_token_kwargs(tokenizer: Any) -> dict[str, Any]:
    """``pad_token_id`` / ``eos_token_id`` / ``bos_token_id`` kwargs for ``GenerationMixin.generate``.

    :class:`GemmaTextEncoderConfigurator` constructs ``Gemma4ForConditionalGeneration`` under ``torch.device('meta')``.
    Some ``transformers`` versions then keep special-token placeholders on ``meta``; ``generate`` compares them with
    ``torch.isin`` and triggers ``Tensor.item()`` on meta tensors. Passing plain ints from the tokenizer avoids that.
    """
    out: dict[str, Any] = {}
    pad = getattr(tokenizer, "pad_token_id", None)
    out["pad_token_id"] = 0 if pad is None else _int_token_id_for_generate(pad)

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple)):
            out["eos_token_id"] = [_int_token_id_for_generate(x) for x in eos]
        else:
            out["eos_token_id"] = _int_token_id_for_generate(eos)

    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        out["bos_token_id"] = _int_token_id_for_generate(bos)

    return out


def _encoder_materialized_device(module: torch.nn.Module) -> torch.device:
    """Device for ``input_ids`` / ``attention_mask`` / ``processor`` batch — must match ``inputs_embeds``.

    ``transformers`` does ``attention_mask = attention_mask.to(device=inputs_embeds.device)`` inside mask
    preprocessing. ``inputs_embeds`` come from ``get_input_embeddings()(input_ids)``, so they live on the **embedding
    weight** device. Inferring CUDA from an unrelated parameter (or from a duplicated ``lm_head`` shard that loaded
    before ``embed_tokens`` in a broken order) desynchronizes masks and triggers ``meta`` scalars in SDPA helpers.

    Use ``get_input_embeddings().weight.device`` whenever that weight exists and is materialized; otherwise fall back
    to any non-``meta`` tensor.
    """
    getter = getattr(module, "get_input_embeddings", None)
    if callable(getter):
        try:
            w = getattr(getter(), "weight", None)
            if w is not None:
                if w.device.type == "meta":
                    msg = (
                        "Gemma ``embed_tokens.weight`` is still on the 'meta' device, so ``inputs_embeds`` would be "
                        "meta and attention masking would fail. Ensure the safetensors loader has materialized "
                        "language-model embedding weights before encode() or generate()."
                    )
                    raise RuntimeError(msg)
                return w.device
        except RuntimeError:
            raise
        except Exception:
            pass

    for t in (*module.parameters(), *module.buffers()):
        if t is not None and t.device.type != "meta":
            return t.device
    msg = (
        "Cannot resolve a non-meta device for Gemma: all parameters and buffers are still on 'meta'. "
        "Finish loading weights onto GPU/CPU before calling encode() or generate()."
    )
    raise RuntimeError(msg)


@contextlib.contextmanager
def _gemma4_attn_eager_for_masking(model: Gemma4ForConditionalGeneration) -> Iterator[None]:
    """Force eager attention while active.

    With ``sdpa`` + ``allow_is_causal_skip=True``, ``transformers`` can call ``padding_mask.all()`` on tensors that
    still live on ``meta`` when the module was meta-initialized (torch 2.x + recent ``transformers``). ``eager_mask``
    delegates to ``sdpa_mask`` with ``allow_is_causal_skip=False``, which avoids that fast path and materializes a
    proper 4-D mask instead.
    """
    configs: list[Any] = [model.config]
    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is not None and text_cfg is not model.config:
        configs.append(text_cfg)
    inner = getattr(model, "model", None)
    lm = getattr(inner, "language_model", None) if inner is not None else None
    lm_cfg = getattr(lm, "config", None) if lm is not None else None
    if lm_cfg is not None and lm_cfg not in configs:
        configs.append(lm_cfg)
    saved: list[tuple[Any, Any]] = []
    for cfg in configs:
        if hasattr(cfg, "_attn_implementation"):
            saved.append((cfg, cfg._attn_implementation))
            cfg._attn_implementation = "eager"
    try:
        yield
    finally:
        for cfg, prev in saved:
            cfg._attn_implementation = prev


def _maybe_save_hf_style_encode_debug(
    tag: str,
    tensors: dict[str, torch.Tensor],
    meta_extra: dict[str, Any],
) -> None:
    """If ``LTX_GEMMA_ENCODE_DEBUG_DIR`` is set, write a small Hub-like folder (``config.json``, ``*.safetensors``, ``encode_meta.json``)."""
    root = (os.environ.get("LTX_GEMMA_ENCODE_DEBUG_DIR") or "").strip()
    if not root:
        return

    from safetensors.torch import save_file

    dump_dir = Path(root).expanduser() / f"{tag}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    cpu_tensors = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    save_file(cpu_tensors, str(dump_dir / "model_input.safetensors"))

    shapes = {k: list(v.shape) for k, v in cpu_tensors.items()}
    meta = {**meta_extra, "tensor_keys": list(cpu_tensors.keys()), "shapes": shapes, "dtypes": {k: str(v.dtype) for k, v in cpu_tensors.items()}}
    (dump_dir / "encode_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (dump_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "ltx_gemma_encode_debug",
                "debug_tag": tag,
                "note": "Not a trainable HF checkpoint; tensor bundle saved for reproducing LTX Gemma encoder inputs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class GemmaTextEncoder(torch.nn.Module):
    """Pure Gemma text encoder — runs the LLM and returns raw hidden states.
    Prompt enhancement (generate) is also supported since the full
    Gemma4ForConditionalGeneration model (including lm_head) is loaded.

    Debug: set ``LTX_GEMMA_ENCODE_DEBUG_DIR`` to a folder path to save each ``encode()`` /
    ``_enhance()`` tensor batch as ``model_input.safetensors`` plus ``config.json`` and ``encode_meta.json``
    (Hub-style filenames for easy inspection; not a runnable HF revision).
    """

    def __init__(
        self,
        model: Gemma4ForConditionalGeneration | None = None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        processor: AutoProcessor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self._dtype = dtype

    def encode(
        self,
        text: str,
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Run Gemma LLM and return raw hidden states + attention mask.
        Calls the inner model (self.model.model) to skip lm_head logits computation (~500 MiB saving).
        Returns:
            (hidden_states, attention_mask) where hidden_states is a tuple of per-layer tensors.
        """
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        dev = _encoder_materialized_device(self.model)
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=dev, dtype=torch.long)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=dev, dtype=torch.long)
        _maybe_save_hf_style_encode_debug(
            "encode",
            {"input_ids": input_ids, "attention_mask": attention_mask},
            {"text_excerpt": text[:4000]},
        )
        with _gemma4_attn_eager_for_masking(self.model):
            outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        del outputs
        return hidden_states, attention_mask

    # --- Prompt enhancement methods ---

    def _enhance(
        self,
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        gen_kw = _generation_special_token_kwargs(self.processor.tokenizer)
        dev = _encoder_materialized_device(self.model)
        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(dev)
        model_inputs = _pad_inputs_for_attention_alignment(
            model_inputs, pad_token_id=int(gen_kw["pad_token_id"])
        )

        tensor_batch = {k: v for k, v in dict(model_inputs).items() if isinstance(v, torch.Tensor)}
        _maybe_save_hf_style_encode_debug(
            "enhance_generate",
            tensor_batch,
            {"chat_template_text_excerpt": text[:4000]},
        )

        fork = torch.random.fork_rng(devices=[dev]) if dev.type == "cuda" else contextlib.nullcontext()
        with torch.inference_mode(), fork, _gemma4_attn_eager_for_masking(self.model):
            torch.manual_seed(seed)
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=min(max_new_tokens, 256),
                do_sample=True,
                temperature=0.45,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                **gen_kw,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for T2V generation."""
        system_prompt = system_prompt or self.default_gemma_t2v_system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]

        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self.default_gemma_i2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed)

    @functools.cached_property
    def default_gemma_i2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_i2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma_t2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_t2v_system_prompt.txt")


# --- Standalone utility functions ---


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}", "r") as f:
        return f.read()


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Concatenate a tensor with a padding tensor of the given value."""
    return torch.cat(
        [
            tensor,
            torch.full(
                (1, padding_length),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=1,
    )


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility."""
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)
        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)
        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs


def _tokenizer_pretrained_dir(gemma_root: str) -> Path:
    """Directory suitable for ``AutoTokenizer.from_pretrained`` (SentencePiece or fast JSON layout)."""
    try:
        return find_matching_file(gemma_root, "tokenizer.model").parent
    except FileNotFoundError:
        return find_matching_file(gemma_root, "tokenizer.json").parent


def _processor_pretrained_dir(gemma_root: str) -> Path:
    """Directory suitable for ``AutoProcessor.from_pretrained`` (legacy preprocessor or Gemma 4 processor)."""
    try:
        return find_matching_file(gemma_root, "preprocessor_config.json").parent
    except FileNotFoundError:
        return find_matching_file(gemma_root, "processor_config.json").parent


def module_ops_from_gemma_root(
    gemma_root: str,
    gemma_hf_config: dict[str, Any] | None = None,
) -> tuple[ModuleOps, ...]:
    tokenizer_root = str(_tokenizer_pretrained_dir(gemma_root))
    processor_root = str(_processor_pretrained_dir(gemma_root))

    cfg = gemma_hf_config if gemma_hf_config is not None else resolve_gemma_checkpoint_config(
        (str(find_matching_file(gemma_root, "model*.safetensors")),)
    )
    encode_max_len = effective_gemma_encode_max_length(cfg)

    def load_tokenizer(module: GemmaTextEncoder) -> GemmaTextEncoder:
        module.tokenizer = LTXVGemmaTokenizer(tokenizer_root, encode_max_len)
        return module

    def load_processor(module: GemmaTextEncoder) -> GemmaTextEncoder:
        if not module.tokenizer:
            raise ValueError("Tokenizer model operation must be performed before processor model operation")
        # Gemma 4 expects a full processor (image / video / audio + tokenizer) from the same release tree.
        module.processor = AutoProcessor.from_pretrained(processor_root, local_files_only=True)
        return module

    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    processor_load_ops = ModuleOps(
        "ProcessorLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.processor is None,
        mutator=load_processor,
    )
    return (tokenizer_load_ops, processor_load_ops)
