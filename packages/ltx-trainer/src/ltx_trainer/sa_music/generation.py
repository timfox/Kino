"""Five-level TELeR prompting for ABC generation (Sec. 3.2, Table 5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TelrPromptLevel:
    level: int
    name: str
    template: str


PROMPT_LEVELS: tuple[TelrPromptLevel, ...] = (
    TelrPromptLevel(
        1,
        "minimum",
        "Generate a melody in ABC notation for the full lyrics {lyrics}.",
    ),
    TelrPromptLevel(
        2,
        "moderate",
        "Generate a melody in ABC notation for the full lyrics {lyrics}, in Scale C with "
        "{time_sig} time signature, without any comments and empty lines. "
        "Indian classical theme about {theme}.",
    ),
    TelrPromptLevel(
        3,
        "moderate_listwise",
        "Generate ABC for lyrics ensuring: (1) Indian classical style, "
        "(2) {genre} Song, (3) Scale C and {time_sig}, (4) notation only, "
        "(5) theme {theme}, ~2-3 minutes. {lyrics}",
    ),
    TelrPromptLevel(
        4,
        "significant",
        "Level 3 + output should be coherent, thematic, stylistically authentic, valid ABC. {lyrics}",
    ),
    TelrPromptLevel(
        5,
        "maximum",
        "Level 4 + raga/tala/gamaka background; monophonic melodic development. {lyrics}",
    ),
)


def build_prompt(
    level: int,
    *,
    lyrics: str,
    theme: str,
    time_sig: str = "3/4",
    genre: str = "Rabindra",
) -> str:
    tpl = next(p for p in PROMPT_LEVELS if p.level == level)
    return tpl.template.format(lyrics=lyrics, theme=theme, time_sig=time_sig, genre=genre)
