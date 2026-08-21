"""LoRA rank index maps from paper Table 2 / Appendix B."""

from __future__ import annotations

# Long-context stress test rank columns r1..r9
LLAMA_LONGCONTEXT_RANKS: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 12, 14, 16)
QWEN_LONGCONTEXT_RANKS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# PhoneBook rank columns p1..p7 (both models)
PHONEBOOK_RANKS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

LONGCONTEXT_LENGTH_BUCKETS: tuple[int, ...] = (
    50,
    100,
    200,
    500,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
)

PHONEBOOK_LENGTH_BUCKETS: tuple[int, ...] = (1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000)


def longcontext_ranks(model: str) -> tuple[int, ...]:
    if "Llama" in model:
        return LLAMA_LONGCONTEXT_RANKS
    return QWEN_LONGCONTEXT_RANKS
