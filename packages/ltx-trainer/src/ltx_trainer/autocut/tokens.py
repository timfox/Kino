"""Unified multimodal token vocabulary (video / audio / text)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.autocut.config import AutoCutConfig, RQVAEConfig
from ltx_trainer.autocut.rqvae import ResidualRQVAE


@dataclass(frozen=True)
class ParsedTokenSequence:
    video_tokens: tuple[str, ...]
    audio_tokens: tuple[str, ...]
    text_blocks: tuple[str, ...]


def special_tokens(cfg: AutoCutConfig | None = None) -> dict[str, str]:
    return {
        "video_start": "<|video_start|>",
        "video_end": "<|video_end|>",
        "audio_start": "<|audio_start|>",
        "audio_end": "<|audio_end|>",
        "text_start": "<|text_start|>",
        "text_end": "<|text_end|>",
        "clip_start": "<|clip_start|>",
        "clip_end": "<|clip_end|>",
    }


def format_video_token(head: int, code: int) -> str:
    return f"<|video_{head}_{code}|>"


def format_audio_token(head: int, code: int) -> str:
    return f"<|audio_{head}_{code}|>"


def embedding_to_video_tokens(
    embedding: Any,
    cfg: RQVAEConfig | None = None,
    *,
    seed: int = 0,
) -> list[str]:
    c = cfg or RQVAEConfig(input_dim=128)
    q = ResidualRQVAE(c, seed=seed).encode(np.asarray(embedding))
    return [format_video_token(h, code) for h, code in q.codes]


def embedding_to_audio_tokens(
    embedding: Any,
    cfg: RQVAEConfig | None = None,
    *,
    seed: int = 0,
) -> list[str]:
    c = cfg or RQVAEConfig(input_dim=2048, codebook_dim=256, encoder_mlp=(1024, 512), decoder_mlp=(512, 1024))
    q = ResidualRQVAE(c, seed=seed).encode(np.asarray(embedding))
    return [format_audio_token(h, code) for h, code in q.codes]


def serialize_alignment_sample(
    *,
    script_blocks: list[str],
    video_token_rows: list[list[str]],
    audio_tokens: list[str],
) -> str:
    """Serialize multimodal alignment training string (Fig. 2)."""
    parts: list[str] = []
    if audio_tokens:
        parts.append("<|audio_start|>")
        parts.extend(audio_tokens)
        parts.append("<|audio_end|>")
    parts.append("<|video_start|>")
    for row in video_token_rows:
        parts.extend(row)
    parts.append("<|video_end|>")
    for i, block in enumerate(script_blocks):
        parts.append("<|text_start|>")
        parts.append(f"Script {i + 1}")
        parts.append(block)
        parts.append("<|text_end|>")
    return "\n".join(parts)


_VIDEO_TOKEN_RE = re.compile(r"<\|video_(\d+)_(\d+)\|>")
_AUDIO_TOKEN_RE = re.compile(r"<\|audio_(\d+)_(\d+)\|>")


def parse_token_ids(text: str) -> ParsedTokenSequence:
    video = tuple(f"<|video_{m.group(1)}_{m.group(2)}|>" for m in _VIDEO_TOKEN_RE.finditer(text))
    audio = tuple(f"<|audio_{m.group(1)}_{m.group(2)}|>" for m in _AUDIO_TOKEN_RE.finditer(text))
    blocks: list[str] = []
    for chunk in text.split("<|text_start|>")[1:]:
        body = chunk.split("<|text_end|>")[0].strip()
        if body:
            blocks.append(body)
    return ParsedTokenSequence(video_tokens=video, audio_tokens=audio, text_blocks=tuple(blocks))


def parse_codes_from_tokens(tokens: Sequence[str], *, prefix: str = "video") -> list[tuple[int, int]]:
    pat = _VIDEO_TOKEN_RE if prefix == "video" else _AUDIO_TOKEN_RE
    out: list[tuple[int, int]] = []
    for t in tokens:
        m = pat.search(t)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def token_indices_from_embedding(
    embedding: Any,
    *,
    num_heads: int = 8,
    codebook_size: int = 256,
    seed: int = 0,
) -> list[tuple[int, int]]:
    """Quantize embedding via RQ-VAE (preferred over legacy RNG stub)."""
    cfg = RQVAEConfig(input_dim=max(4, int(np.asarray(embedding).size)), quant_heads=num_heads, codebook_size=codebook_size)
    q = ResidualRQVAE(cfg, seed=seed).encode(np.asarray(embedding))
    return q.codes
