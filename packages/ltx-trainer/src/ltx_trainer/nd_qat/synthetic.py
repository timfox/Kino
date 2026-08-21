"""SPECK32/64 stub and ciphertext-pair dataset (Sec. 2)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.nd_qat.config import INPUT_CHANNELS, INPUT_HEIGHT, INPUT_DIFF_HEX, GROUP_SIZE


def _xor16(a: Tensor, b: Tensor) -> Tensor:
    return (a ^ b).to(torch.int64)


def speck32_encrypt_pair(
    p0: Tensor,
    p1: Tensor,
    key: Tensor,
    *,
    rounds: int = 6,
) -> tuple[Tensor, Tensor]:
    """Minimal SPECK32/64 round stub for synthetic labels (not cryptographically exact)."""
    x = p0.clone()
    y = p1.clone()
    k = key
    for r in range(rounds):
        x = _xor16(x, ((y << 2) | (y >> 14)) & 0xFFFF)
        x = (x + k) & 0xFFFF
        y = _xor16(y, ((x << 7) | (x >> 9)) & 0xFFFF)
        y = (y + r) & 0xFFFF
        k = _xor16(k, ((k << 1) | (k >> 15)) & 0xFFFF)
    return x, y


def _word_bits(word: Tensor) -> Tensor:
    shifts = torch.arange(16, device=word.device, dtype=word.dtype)
    return ((word.unsqueeze(-1) >> shifts) & 1).float()


def ciphertext_to_feature(cl: Tensor, cr: Tensor, cpl: Tensor, cpr: Tensor) -> Tensor:
    """4×16 binary matrix (Cl, Cr, C'l, C'r) per sample → (4, 16)."""
    return torch.stack([_word_bits(cl), _word_bits(cr), _word_bits(cpl), _word_bits(cpr)], dim=0)


def group_samples(x: Tensor, y: Tensor, *, group: int = GROUP_SIZE) -> tuple[Tensor, Tensor]:
    """Group 8 same-label samples → 4×16×8 tensor (Sec. 2)."""
    n = x.shape[0] // group * group
    x = x[:n].view(-1, group, *x.shape[1:])
    y = y[:n].view(-1, group)[:, 0]
    return x, y


def synthetic_batch(
    batch: int = 32,
    *,
    rounds: int = 6,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Generate binary ciphertext-pair features and real/random labels.

    Real: fixed input difference; random: second plaintext resampled.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    key = torch.randint(0, 65536, (batch,), generator=gen, dtype=torch.int64)
    p0 = torch.randint(0, 65536, (batch,), generator=gen, dtype=torch.int64)
    diff = 0x0040
    p1 = (p0 ^ diff) & 0xFFFF
    labels = torch.randint(0, 2, (batch,), generator=gen)
    p1_rand = torch.randint(0, 65536, (batch,), generator=gen, dtype=torch.int64)
    p1 = torch.where(labels.bool(), p1, p1_rand)
    c0, c1 = speck32_encrypt_pair(p0, p1, key, rounds=rounds)
    c0p, c1p = speck32_encrypt_pair(p0 ^ 1, p1 ^ 1, key, rounds=rounds)
    feats = []
    for i in range(batch):
        feats.append(ciphertext_to_feature(c0[i], c1[i], c0p[i], c1p[i]))
    x = torch.stack(feats)  # (B, 4, 16)
    if GROUP_SIZE > 1:
        x = x.unsqueeze(-1).expand(-1, -1, -1, GROUP_SIZE)
    return x, labels.float()


def dataset_summary() -> dict:
    return {
        "cipher": "SPECK32/64",
        "rounds": 6,
        "input_difference": INPUT_DIFF_HEX,
        "samples_per_class": 5_000_000,
        "input_shape": [4, 16, 8],
        "label_threshold": 0.505,
    }
