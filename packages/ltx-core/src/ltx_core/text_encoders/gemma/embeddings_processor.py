from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector


def _max_learnable_register_stride(*connectors: Embeddings1DConnector | None) -> int:
    """``Embeddings1DConnector`` requires ``seq_len % num_learnable_registers == 0`` when registers are enabled."""
    stride = 1
    for c in connectors:
        if c is None:
            continue
        n = getattr(c, "num_learnable_registers", None)
        if n:
            stride = max(stride, int(n))
    return stride


def _pad_seq_to_stride(
    video_feats: torch.Tensor,
    audio_feats: torch.Tensor | None,
    attention_mask: torch.Tensor,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Right-pad sequence dim so connectors' learnable-register tiling can run (default stride 128)."""
    if stride <= 1:
        return video_feats, audio_feats, attention_mask
    t = int(video_feats.shape[1])
    if t % stride == 0:
        return video_feats, audio_feats, attention_mask
    pad_len = stride * ((t + stride - 1) // stride) - t
    # [B, T, C] — pad T on the right; mask — pad last index dimension with 0 (masked / padding tokens).
    video_feats = F.pad(video_feats, (0, 0, 0, pad_len))
    if audio_feats is not None:
        audio_feats = F.pad(audio_feats, (0, 0, 0, pad_len))
    attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
    return video_feats, audio_feats, attention_mask


class EmbeddingsProcessorOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor | None
    attention_mask: torch.Tensor


def convert_to_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert binary attention mask to additive form for transformer masking."""
    return (attention_mask.to(torch.int64) - 1).to(dtype).reshape(
        (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
    ) * torch.finfo(dtype).max


def _to_binary_mask(encoded: torch.Tensor, encoded_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert connector output mask to binary mask and apply to encoded tensor."""
    binary_mask = (encoded_mask < 0.000001).to(torch.int64)
    binary_mask = binary_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
    encoded = encoded * binary_mask
    return encoded, binary_mask


class EmbeddingsProcessor(nn.Module):
    """Wraps feature extractor + video connector + optional audio connector.
    Can operate in two modes:
    1. create_embeddings(): Takes pre-computed features + additive mask (backward compat, used by trainer)
    2. process_hidden_states(): Takes raw Gemma hidden states, runs feature extraction + connectors
    """

    def __init__(
        self,
        *,
        feature_extractor: nn.Module | None = None,
        video_connector: Embeddings1DConnector,
        audio_connector: Embeddings1DConnector | None = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.video_connector = video_connector
        self.audio_connector = audio_connector

    def create_embeddings(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor | None,
        additive_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.audio_connector is not None and audio_features is None:
            raise ValueError("Audio connector is configured but no audio features were provided.")
        if self.audio_connector is None and audio_features is not None:
            raise ValueError("Audio features were provided but no audio connector is configured.")

        stride = _max_learnable_register_stride(self.video_connector, self.audio_connector)
        if stride > 1:
            t_vid = int(video_features.shape[1])
            if t_vid % stride != 0:
                pad_len = stride * ((t_vid + stride - 1) // stride) - t_vid
                video_features = F.pad(video_features, (0, 0, 0, pad_len))
                if audio_features is not None:
                    audio_features = F.pad(audio_features, (0, 0, 0, pad_len))
                # Padded positions must read as invalid to the connector (same as ``convert_to_additive_mask``).
                pad_val = -torch.finfo(additive_attention_mask.dtype).max
                additive_attention_mask = F.pad(additive_attention_mask, (0, pad_len), value=pad_val)

        video_encoded, video_mask = self.video_connector(video_features, additive_attention_mask)
        video_encoded, binary_mask = _to_binary_mask(video_encoded, video_mask)

        audio_encoded = None
        if self.audio_connector is not None:
            audio_encoded, _ = self.audio_connector(audio_features, additive_attention_mask)

        return video_encoded, audio_encoded, binary_mask.squeeze(-1)

    def process_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ) -> EmbeddingsProcessorOutput:
        """Full pipeline: feature extraction -> connectors -> final embeddings.
        Args:
            hidden_states: Raw Gemma hidden states (tuple of tensors per layer).
            attention_mask: Binary attention mask [B, seq_len].
            padding_side: Padding side used during tokenization.
        Returns:
            EmbeddingsProcessorOutput with video_encoding, audio_encoding, and attention_mask.
        """
        if self.feature_extractor is None:
            raise ValueError("feature_extractor is required for process_hidden_states()")

        video_feats, audio_feats = self.feature_extractor(hidden_states, attention_mask, padding_side)
        stride = _max_learnable_register_stride(self.video_connector, self.audio_connector)
        video_feats, audio_feats, attention_mask = _pad_seq_to_stride(
            video_feats, audio_feats, attention_mask, stride
        )
        additive_mask = convert_to_additive_mask(attention_mask, video_feats.dtype)
        video_enc, audio_enc, binary_mask = self.create_embeddings(video_feats, audio_feats, additive_mask)
        return EmbeddingsProcessorOutput(video_enc, audio_enc, binary_mask)
