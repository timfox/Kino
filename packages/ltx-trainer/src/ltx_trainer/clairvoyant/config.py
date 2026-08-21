"""Clairvoyant sidecar proxy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.clairvoyant.constants import (
    MEDIUM_MAX_TOKENS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    SHORT_MAX_TOKENS,
    STARVATION_TAU_MULTIPLIER,
    UPSTREAM_REPO,
)


@dataclass
class ClairvoyantConfig:
    """OpenAI-compatible serial-backend sidecar settings."""

    listen_host: str = "127.0.0.1"
    listen_port: int = 8080
    upstream_url: str = "http://127.0.0.1:11434"
    model_variant: str = "sharegpt"  # sharegpt | lmsys | oasst1
    mu_short_s: float = 3.5  # mean short sojourn under queueing (RTX 4090 default)
    starvation_tau_s: float | None = None  # default 3 × mu_short
    short_max_tokens: int = SHORT_MAX_TOKENS
    medium_max_tokens: int = MEDIUM_MAX_TOKENS

    def __post_init__(self) -> None:
        if self.starvation_tau_s is None:
            self.starvation_tau_s = STARVATION_TAU_MULTIPLIER * self.mu_short_s

    @property
    def tau_multiplier(self) -> float:
        if self.mu_short_s <= 0:
            return STARVATION_TAU_MULTIPLIER
        return float(self.starvation_tau_s or 0) / self.mu_short_s


__all__ = [
    "ClairvoyantConfig",
    "MEDIUM_MAX_TOKENS",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "SHORT_MAX_TOKENS",
    "UPSTREAM_REPO",
]
