"""FIFO-scheduled async streaming inference stub (Algorithm 3)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ltx_trainer.audio_interaction.tokens import StreamStep, StreamToken


@dataclass
class FIFOScheduler:
    """Encoder producer + decoder consumer with drain-on-trigger."""

    queue: deque = field(default_factory=deque)
    last_token: StreamToken = "silent"
    steps: list[StreamStep] = field(default_factory=list)

    def encode_append(self, chunk_idx: int, feature_norm: float) -> None:
        self.queue.append((chunk_idx, feature_norm))

    def decoder_tick(self, respond_score: float, threshold: float = 0.5) -> StreamStep | None:
        if self.last_token in ("silent", "eos"):
            if not self.queue:
                return None
            flushed = list(self.queue)
            self.queue.clear()
            _ = flushed
            token: StreamToken = "response" if respond_score >= threshold else "silent"
            step = StreamStep(chunk_idx=flushed[-1][0], token=token)
            self.last_token = token
            self.steps.append(step)
            return step
        # mid-response: autoregressive text step
        step = StreamStep(chunk_idx=self.steps[-1].chunk_idx, token="text", text="…")
        self.last_token = "eos"
        self.steps.append(StreamStep(chunk_idx=step.chunk_idx, token="eos"))
        return step
