"""Perceive–decide–respond loop stub: (dt, rt) = f(a≤t, d<t, r<t)."""

from __future__ import annotations

import torch

from ltx_trainer.audio_interaction.config import AudioInteractionConfig
from ltx_trainer.audio_interaction.fifo import FIFOScheduler
from ltx_trainer.audio_interaction.tfjp import tfjp_preprocess


def chunk_waveform(waveform: torch.Tensor, chunk_samples: int) -> list[torch.Tensor]:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    t = waveform.shape[-1]
    chunks: list[torch.Tensor] = []
    for start in range(0, t, chunk_samples):
        end = min(start + chunk_samples, t)
        pad = chunk_samples - (end - start)
        c = waveform[..., start:end]
        if pad > 0:
            c = torch.nn.functional.pad(c, (0, pad))
        chunks.append(c)
    return chunks


def respond_score_from_chunk(chunk: torch.Tensor) -> float:
    """Stub: high energy + high zero-crossing rate → respond."""
    e = chunk.float().abs().mean().item()
    zc = ((chunk[..., 1:] * chunk[..., :-1]) < 0).float().mean().item()
    return min(1.0, 0.35 * e + 0.65 * zc)


def streaming_interaction_loop(
    waveform: torch.Tensor,
    cfg: AudioInteractionConfig | None = None,
    *,
    sample_rate: int = 16_000,
) -> dict:
    cfg = cfg or AudioInteractionConfig()
    chunk_samples = int(sample_rate * cfg.chunk_ms / 1000)
    cleaned = tfjp_preprocess(waveform)
    chunks = chunk_waveform(cleaned, chunk_samples)
    sched = FIFOScheduler()
    decisions = []
    for i, ch in enumerate(chunks):
        score = respond_score_from_chunk(ch)
        sched.encode_append(i, score)
        step = sched.decoder_tick(score)
        if step is not None:
            decisions.append(step.to_dict())
    return {
        "n_chunks": len(chunks),
        "chunk_ms": cfg.chunk_ms,
        "decisions": decisions,
        "fifo_steps": [s.to_dict() for s in sched.steps],
    }
