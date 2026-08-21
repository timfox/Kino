"""Streaming control tokens (Section 3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StreamToken = Literal["silent", "response", "eos", "text"]


@dataclass
class StreamStep:
    chunk_idx: int
    token: StreamToken
    text: str = ""

    def to_dict(self) -> dict:
        return {"chunk_idx": self.chunk_idx, "token": self.token, "text": self.text}
