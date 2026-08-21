"""Clean/noisy routing mask for KV-cacheable LMDM inference (Sec. 4.1, Eq. 4–5)."""

from __future__ import annotations


def routing_mask(context_frames: int, target_frames: int) -> list[int]:
    r"""r := [0_{1:s}, 1_{s:s+o}] — route noisy latents only on target block."""
    if context_frames < 0 or target_frames < 0:
        raise ValueError("frame counts must be non-negative")
    return [0] * context_frames + [1] * target_frames


def routed_noisy_latent(noisy: float, *, masked: bool) -> float:
    """Toy scalar: r ⊙ x(k) zeroes context frames before projection."""
    return noisy if masked else 0.0


def context_hidden_independent_of_noise(context_value: float, *, w_clean: float) -> float:
    r"""h^{1:s} = B·x_clean — independent of noise level k (Eq. 5)."""
    return w_clean * context_value
