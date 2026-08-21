"""Mini benchmark examples (paper Fig. 1 / Sec. 6.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    scene: str
    speech: str
    setting: str = "sync_av"

    @property
    def prompt(self) -> str:
        return f"{self.scene} <S>{self.speech}<E>"


BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "sydney_harbour",
        "A woman stands on the harbour's edge at twilight, the sails of the Sydney "
        "Opera House catching the last apricot glow of sunset.",
        "Some buildings aren't made of stone, but of gathered breath and gathered dreams.",
    ),
    BenchmarkCase(
        "shanghai_lny",
        "A man stands on a bustling street in Shanghai during Chinese Lunar New Year, "
        "red lanterns hanging overhead.",
        "Wishing everyone a Happy New Year and joy every single day.",
    ),
    BenchmarkCase(
        "beach_potential",
        "A woman in a white shirt stands on golden beach sands, waves rolling beyond her.",
        "Your potential is infinite, so never give up.",
    ),
    BenchmarkCase(
        "openai_podium",
        "A man stands at a podium in a conference room with profit charts on a screen.",
        "The board wants to sell OpenAI to Zuckerberg, which is unacceptable.",
    ),
    BenchmarkCase(
        "trevi_fountain",
        "A woman stands before the Trevi Fountain at dawn, holding two coins.",
        "May your wishes find their way to water, and your heart always know the way back.",
    ),
)


def case_by_id(case_id: str) -> BenchmarkCase | None:
    for c in BENCHMARK_CASES:
        if c.case_id == case_id:
            return c
    return None
