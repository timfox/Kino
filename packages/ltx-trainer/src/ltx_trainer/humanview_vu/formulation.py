"""Unified watch–remember–reason formulation (Sec. 2.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoUnderstandingState:
    """Toy state for F_VU = (F_watch, F_remember, F_reason, F_out)."""

    num_frames: int = 8
    memory_dim: int = 4
    query: str = "When does the person enter the room?"
    perceptual: list[float] = field(default_factory=list)
    memory: list[float] = field(default_factory=list)
    reasoning_trace: list[str] = field(default_factory=list)
    output: str | None = None


def f_watch(num_frames: int, query: str, *, select_ratio: float = 0.5) -> list[float]:
    """Query-aware frame/token selection proxy (Eq. 1)."""
    k = max(1, int(num_frames * select_ratio))
    base = [float(i) / max(num_frames - 1, 1) for i in range(num_frames)]
    return base[:k]


def f_remember(prev: list[float], z_t: float, *, decay: float = 0.85) -> list[float]:
    """Memory update with exponential decay (Eq. 2)."""
    if not prev:
        return [z_t]
    updated = [decay * v for v in prev]
    updated.append(z_t)
    return updated[-8:]


def f_reason(perceptual: list[float], memory: list[float], query: str) -> list[str]:
    """Textual reasoning trace proxy (Eq. 3)."""
    if not perceptual:
        return [f"No perceptual evidence for query: {query}"]
    t_peak = perceptual[-1]
    mem_span = (memory[0], memory[-1]) if len(memory) >= 2 else (memory[0], memory[0])
    return [
        f"Scan frames around t={t_peak:.2f}",
        f"Recall memory span [{mem_span[0]:.2f}, {mem_span[1]:.2f}]",
        "Integrate multimodal cues before answering",
    ]


def f_out(reasoning: list[str], query: str) -> str:
    """Final prediction (Eq. 4)."""
    return f"Answer({query!r}) after {len(reasoning)} reasoning steps"


def run_formulation_demo(num_frames: int = 12, query: str | None = None) -> dict[str, Any]:
    q = query or "What happens after the door opens?"
    z_seq = f_watch(num_frames, q, select_ratio=0.4)
    memory: list[float] = []
    for z in z_seq:
        memory = f_remember(memory, z)
    reasoning = f_reason(z_seq, memory, q)
    output = f_out(reasoning, q)
    return {
        "query": q,
        "num_frames": num_frames,
        "selected_timesteps": z_seq,
        "memory_tail": memory[-4:],
        "reasoning_trace": reasoning,
        "output": output,
        "training_paradigms": ["SFT", "GRPO"],
    }


def sft_loss_proxy(log_probs: list[float]) -> float:
    return -sum(log_probs) / max(len(log_probs), 1)


def grpo_group_normalize(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / max(len(rewards), 1)
    std = var**0.5 or 1.0
    return [(r - mean) / std for r in rewards]
