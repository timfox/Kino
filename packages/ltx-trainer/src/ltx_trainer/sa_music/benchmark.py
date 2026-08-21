"""504-question South Asian music understanding benchmark (Sec. 3.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSubtask:
    name: str
    questions: int
    description: str


SUBTASKS: tuple[BenchmarkSubtask, ...] = (
    BenchmarkSubtask(
        "music_theory_understanding",
        163,
        "Raga grammar, tala, thaat, ornamentation (Hindustani/Bengali)",
    ),
    BenchmarkSubtask(
        "music_general_knowledge",
        143,
        "Composers, instruments, regional traditions",
    ),
    BenchmarkSubtask(
        "music_continuation",
        198,
        "ABC notation continuation from Rabindra/Nazrul seeds",
    ),
)


def total_questions() -> int:
    return sum(s.questions for s in SUBTASKS)


def extract_mcq_answer(response: str) -> str | None:
    """Appendix C: explicit [[Answer: X]] then standalone letter counting."""
    import re

    text = response.strip()
    explicit = re.search(r"\[\[Answer:\s*([A-D])\]\]", text, re.I)
    if explicit:
        return explicit.group(1).upper()
    for pat in (
        r"(?:correct answer is|option)\s*([A-D])\b",
        r"\b([A-D])\s*is correct",
    ):
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1).upper()
    letters = re.findall(r"\b([A-D])\b", text)
    if not letters:
        return None
    counts: dict[str, int] = {}
    for ch in letters:
        counts[ch] = counts.get(ch, 0) + 1
    top = max(counts.values())
    winners = [k for k, v in counts.items() if v == top]
    if len(winners) == 1:
        return winners[0]
    return None


def accuracy(predictions: list[str | None], gold: list[str]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, gold, strict=True) if p == g)
    return correct / len(gold)
