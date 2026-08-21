"""MMAE configuration and taxonomy enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ltx_trainer.mmae.constants import (
    MMAE_GITHUB,
    MMAE_HUB_DATASET,
    MMAE_JUDGER,
    MMAE_PAPER_ARXIV,
    MMAE_PAPER_TITLE,
    MMAE_TOTAL_SAMPLES,
)


class Modality(str, Enum):
    SOUND = "sound"
    MUSIC = "music"
    SPEECH = "speech"
    SOUND_MUSIC = "sound-music"
    SOUND_SPEECH = "sound-speech"
    MUSIC_SPEECH = "music-speech"
    SOUND_MUSIC_SPEECH = "sound-music-speech"
    MIX = "mix"


class Complexity(str, Enum):
    SINGLE = "single"
    MULTI_PART = "multi-part"
    MULTI_INSTRUCTION = "multi-instruction"
    MULTI_AUDIO = "multi-audio"
    MULTI_ROUND = "multi-round"
    MULTI_HOP = "multi-hop"


class Granularity(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"


class RubricCategory(str, Enum):
    INSTRUCTION_FOLLOWING = "Instruction Following"
    CONSISTENCY = "Consistency"


class LocalOperation(str, Enum):
    ADDITION = "addition"
    REMOVAL = "removal"
    REPLACEMENT = "replacement"
    EXTRACTION = "extraction"
    ALTERATION = "alteration"


class GlobalOperation(str, Enum):
    BACKGROUND_CHANGE = "background change"
    FOREGROUND_CHANGE = "foreground change"
    ALTERATION = "alteration"


@dataclass(frozen=True)
class MMAEEvalParams:
    majority_votes: int = 3
    majority_threshold: int = 2
    shuffle_choices: bool = True
    short_max_duration_sec: float = 10.0


@dataclass
class MMAEConfig:
    paper_arxiv: str = MMAE_PAPER_ARXIV
    paper_title: str = MMAE_PAPER_TITLE
    github: str = MMAE_GITHUB
    hub_dataset: str = MMAE_HUB_DATASET
    judger: str = MMAE_JUDGER
    num_samples: int = MMAE_TOTAL_SAMPLES
    params: MMAEEvalParams = field(default_factory=MMAEEvalParams)
    candidate_models: tuple[str, ...] = (
        "Step-Audio-EditX",
        "Ming-UniAudio",
        "MMEdit",
        "Audio-Omni",
        "SmartDJ w/o planner",
        "SmartDJ w/ planner",
        "Identity",
        "Noise",
    )
