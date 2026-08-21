"""Runtime configuration for midsize GPU division projections."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.midint_division.constants import (
    BIT_EXP_MAX,
    BIT_EXP_MIN,
    DEFAULT_SEQUENTIALIZATION_Q,
    DEFAULT_WORD_BITS,
)


@dataclass
class MidintDivisionConfig:
    """Knobs for cost-model and demo runs (CPU reference + CUDA mapping cards)."""

    word_bits: int = DEFAULT_WORD_BITS
    sequentialization_q: int = DEFAULT_SEQUENTIALIZATION_Q
    bits_exp: int = BIT_EXP_MAX
    insts_exp: int = BIT_EXP_MAX - 4  # NumBits · NumInsts = 2^32
    base_b: int = 10
    use_newton_shinv: bool = True
    validate_examples: tuple[dict[str, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not (BIT_EXP_MIN <= self.bits_exp <= BIT_EXP_MAX):
            raise ValueError(f"bits_exp must be in [{BIT_EXP_MIN}, {BIT_EXP_MAX}]")
        if self.bits_exp + self.insts_exp != 32:
            raise ValueError("bits_exp + insts_exp must equal 32 (paper batch constraint)")
