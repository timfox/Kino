import functools
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.text_encoders.gemma.config import effective_gemma_encode_max_length, resolve_gemma_checkpoint_config
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.utils import find_matching_file


class GemmaTextEncoder(torch.nn.Module):
    """Pure Gemma text encoder — runs the LLM and returns raw hidden states.
    Prompt enhancement (generate) is also supported since the full
    Gemma3ForConditionalGeneration model (including lm_head) is loaded.
    """

    def __init__(
        self,
        model: Gemma3ForConditionalGeneration | None = None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        processor: Any | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self._dtype = dtype

    def encode(
        self,
        prompts: list[str],
        padding_side: str = "left",  # noqa: ARG002
    ) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor]]:
        """Run a single fused Gemma forward over a batch of prompts.
        Calls the inner model (self.model.model) to skip lm_head logits computation
        (~500 MiB saving). The tokenizer pads every prompt to ``max_length`` (1024),
        so the inputs stack into a single ``[N, 1024]`` batch with no further padding
        logic; per-prompt outputs are sliced back to ``[1, 1024, D]`` / ``[1, 1024]``
        in the original order.
        """
        if not prompts:
            return []
        tokenized = [self.tokenizer.tokenize_with_weights(t)["gemma"] for t in prompts]
        max_len = max(len(pairs) for pairs in tokenized)
        pad_id = int(self.tokenizer.tokenizer.pad_token_id or 0)

        def _pad_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
            pad_count = max_len - len(pairs)
            if pad_count <= 0:
                return pairs
            # Gemma chat/encode uses left padding so real tokens stay right-aligned.
            return [(pad_id, 0)] * pad_count + list(pairs)

        tokenized = [_pad_pairs(pairs) for pairs in tokenized]
        input_ids = torch.tensor(
            [[tok for tok, _ in pairs] for pairs in tokenized],
            device=self.model.device,
        )
        attention_mask = torch.tensor(
            [[w for _, w in pairs] for pairs in tokenized],
            device=self.model.device,
        )
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        del outputs
        return [(tuple(h[i : i + 1] for h in hidden_states), attention_mask[i : i + 1]) for i in range(len(prompts))]

    # --- Prompt enhancement methods ---

    def _enhance(
        self,
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        with torch.inference_mode(), torch.random.fork_rng(devices=[self.model.device]):
            torch.manual_seed(seed)
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
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
