"""HoloFair prompt templates and SCM triggers (Sec. 3.1.2, Appendix C.3.2)."""

from __future__ import annotations

from ltx_trainer.holofair.config import HoloFairConfig

NEUTRAL_FRAMINGS: tuple[str, ...] = (
    "a close-up photo of a person",
    "a head and shoulders portrait of a person",
    "a clear front-facing portrait of a person",
)

PROMPT_TEMPLATE = (
    "a close-up photo of a {subject} person, photorealistic style, "
    "lit with soft natural light, against a plain studio background, with a neutral expression"
)


def neutral_prompts(*, cfg: HoloFairConfig | None = None) -> list[str]:
    cfg = cfg or HoloFairConfig()
    return list(NEUTRAL_FRAMINGS)


def semantic_prompts(*, cfg: HoloFairConfig | None = None) -> list[str]:
    cfg = cfg or HoloFairConfig()
    return [PROMPT_TEMPLATE.format(subject=trigger) for trigger in cfg.semantic_triggers]


def biased_gen_prompt(
    *,
    race: str,
    age: str,
    gender: str,
) -> str:
    """Gen-set style: explicit demographic attribute (Sec. 3.1.2)."""
    return (
        f"a close-up photo of a {age} {race} {gender} person, "
        "high-resolution photograph, plain studio background"
    )
