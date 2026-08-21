"""Weight page table: Any-Precision bit-plane pages per expert linear-block."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ltx_trainer.pagedweight.config import LINEAR_BLOCKS, SUPPORTED_BITWIDTHS

PageState = Literal["gpu", "cpu", "transfer"]


@dataclass
class LinearBlockPage:
    """One expert linear-block (gate_up or down) with committed/desired bitwidths."""

    layer: int
    expert: int
    block: str  # gate_up | down
    committed_bits: int = 8
    desired_bits: int = 8
    supported: tuple[int, ...] = SUPPORTED_BITWIDTHS
    # memory bytes at each bitwidth (stub: proportional to bits)
    mem_at: dict[int, int] = field(default_factory=dict)
    page_state: PageState = "gpu"
    sensitivity: dict[int, float] = field(default_factory=dict)  # s^b
    routing_mass: float = 0.0

    def __post_init__(self) -> None:
        if not self.mem_at:
            base = 8 * 1024 * 1024  # 8 MiB at 8-bit stub
            self.mem_at = {b: int(base * b / 8) for b in self.supported}
        if not self.sensitivity:
            # lower bits → higher sensitivity damage
            self.sensitivity = {b: float(9 - b) * 0.1 for b in self.supported}

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.layer, self.expert, self.block)

    def released_bytes(self, from_b: int, to_b: int) -> int:
        if to_b >= from_b:
            return 0
        return max(0, self.mem_at.get(from_b, 0) - self.mem_at.get(to_b, 0))

    def global_damage(self, from_b: int, to_b: int) -> float:
        if to_b >= from_b:
            return 0.0
        return max(0.0, self.sensitivity.get(to_b, 0.0) - self.sensitivity.get(from_b, 0.0))


@dataclass
class WeightPageTable:
    """PagedWeight page table over MoE linear-blocks."""

    pages: dict[tuple[int, int, str], LinearBlockPage] = field(default_factory=dict)

    @classmethod
    def build(cls, n_layers: int, n_experts: int, start_bits: int = 8) -> WeightPageTable:
        table = cls()
        for ell in range(n_layers):
            for e in range(n_experts):
                # uneven routing mass: hot experts early
                mass = 1.0 / (1.0 + e)
                for u in LINEAR_BLOCKS:
                    page = LinearBlockPage(
                        layer=ell,
                        expert=e,
                        block=u,
                        committed_bits=start_bits,
                        desired_bits=start_bits,
                        routing_mass=mass,
                    )
                    table.pages[page.key] = page
        return table

    def gpu_resident_bytes(self) -> int:
        return sum(p.mem_at.get(p.committed_bits, 0) for p in self.pages.values() if p.page_state == "gpu")

    def set_desired(self, key: tuple[int, int, str], bits: int) -> None:
        page = self.pages[key]
        if bits in page.supported:
            page.desired_bits = bits
