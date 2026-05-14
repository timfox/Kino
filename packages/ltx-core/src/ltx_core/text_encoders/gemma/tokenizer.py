from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _fallback_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """One token Gemma can run a forward on; HF returns length-0 ``input_ids`` for ``\"\"``."""
    for cand in (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        getattr(tokenizer, "unk_token_id", None),
    ):
        if cand is None:
            continue
        if isinstance(cand, (list, tuple)):
            if not cand:
                continue
            cand = cand[0]
        return int(cand)
    return 0


class LTXVGemmaTokenizer:
    """
    Tokenizer wrapper for Gemma models compatible with LTXV processes.
    This class wraps HuggingFace's `AutoTokenizer` for use with Gemma text encoders,
    ensuring correct settings and output formatting for downstream consumption.
    """

    def __init__(self, tokenizer_path: str, max_length: int = 256):
        """
        Initialize the tokenizer.
        Args:
            tokenizer_path (str): Path to the pretrained tokenizer files or model directory.
            max_length (int, optional): Upper bound on sequence length (truncation only). Encoding pads to the
                actual token count, not to this length, so VRAM scales with prompt size rather than always
                paying the cost of a full-length forward pass.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, model_max_length=max_length
        )
        # Gemma expects left padding for chat-style prompts; for plain text it doesn't matter much.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict[str, list[tuple[int, int]]]:
        """
        Tokenize the given text and return token IDs and attention weights.
        Args:
            text (str): The input string to tokenize.
            return_word_ids (bool, optional): If True, includes the token's position (index) in the output tuples.
                                              If False (default), omits the indices.
        Returns:
            dict[str, list[tuple[int, int]]] OR dict[str, list[tuple[int, int, int]]]:
                A dictionary with a "gemma" key mapping to:
                    - a list of (token_id, attention_mask) tuples if return_word_ids is False;
                    - a list of (token_id, attention_mask, index) tuples if return_word_ids is True.
        Example:
            >>> tokenizer = LTXVGemmaTokenizer("path/to/tokenizer", max_length=8)
            >>> tokenizer.tokenize_with_weights("hello world")
            {'gemma': [(1234, 1), (5678, 1), (2, 0), ...]}
        """
        text = text.strip()
        if not text:
            # Hugging Face Gemma tokenizers return ``input_ids`` shape ``(1, 0)`` for ``""``, which breaks the
            # language model (query reshape with sequence length 0). Whitespace-only prompts become ``""`` here.
            tid = _fallback_token_id(self.tokenizer)
            tuples = [(tid, 1, 0)]
            out = {"gemma": tuples}
            if not return_word_ids:
                out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}
            return out

        # Pad only to the real sequence length (longest in batch). ``padding="max_length"`` would run every
        # forward at ``max_length`` tokens (e.g. 8192), which explodes VRAM for Gemma 4 26B with eager attention.
        encoded = self.tokenizer(
            text,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        if input_ids.shape[1] == 0:
            tid = _fallback_token_id(self.tokenizer)
            tuples = [(tid, 1, 0)]
            out = {"gemma": tuples}
            if not return_word_ids:
                out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}
            return out

        tuples = [
            (token_id, attn, i) for i, (token_id, attn) in enumerate(zip(input_ids[0], attention_mask[0], strict=True))
        ]
        out = {"gemma": tuples}

        if not return_word_ids:
            # Return only (token_id, attention_mask) pairs, omitting token position
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}

        return out
