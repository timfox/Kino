"""ChildVox dataset registry (Table 1)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.childvox.config import ChildVoxCategory


def dataset_registry() -> list[dict[str, Any]]:
    """Summary of 17 child-centered corpora and evaluation tasks."""
    return [
        {"name": "CirCor", "category": ChildVoxCategory.PHYSIOLOGICAL, "task": "murmur_detection", "labels": ["Absent", "Present", "Unknown"], "license": "Open"},
        {"name": "ICBHI", "category": ChildVoxCategory.PHYSIOLOGICAL, "task": "crackles/wheezes/condition", "labels": ["Crackles", "Wheezes", "Healthy/COPD/Other"], "license": "Open"},
        {"name": "SPRSound", "category": ChildVoxCategory.PHYSIOLOGICAL, "task": "respiratory_sound", "labels": ["Normal", "CAS", "DAS", "CAS&DAS", "Poor Quality"], "license": "Open"},
        {"name": "Donate-a-cry", "category": ChildVoxCategory.VOCALIZATION, "task": "cry_cause", "labels": ["Hunger", "Other"], "license": "Open"},
        {"name": "CryBank", "category": ChildVoxCategory.VOCALIZATION, "task": "cry_cause", "labels": ["Hunger", "Loneliness", "Discomfort"], "license": "Not Specified"},
        {"name": "AudioSet-Child", "category": ChildVoxCategory.VOCALIZATION, "task": "child_sound", "labels": 10, "license": "CC-BY-4.0"},
        {"name": "ReCANVo", "category": ChildVoxCategory.CANONICAL_SYLLABLES, "task": "affective_status", "labels": 6, "license": "Not Specified"},
        {"name": "BabbleCor", "category": ChildVoxCategory.CANONICAL_SYLLABLES, "task": "vocal_development", "labels": 5, "license": "Customized"},
        {"name": "SpeechMaturity", "category": ChildVoxCategory.CANONICAL_SYLLABLES, "task": "vocal_development", "labels": 5, "license": "Customized"},
        {"name": "C-BESD", "category": ChildVoxCategory.SPEECH, "task": "emotion", "labels": ["Anger", "Happy", "Neutral", "Sad"], "license": "Not Specified"},
        {"name": "PERCEPT-R", "category": ChildVoxCategory.SPEECH, "task": "rhotic", "labels": ["Rhotic", "Derhotic"], "license": "PhonBank"},
        {"name": "SpeechOcean762", "category": ChildVoxCategory.SPEECH, "task": "prosody/fluency/accuracy", "labels": 3, "license": "CC-BY-4.0"},
        {"name": "UltraSuite", "category": ChildVoxCategory.SPEECH, "task": "articulator", "labels": 8, "license": "CC-BY-NC-4.0"},
        {"name": "NLS", "category": ChildVoxCategory.SPEECH, "task": "diarization/intelligibility", "labels": "private", "license": "Private"},
        {"name": "ADOS2-Mod3", "category": ChildVoxCategory.SPEECH, "task": "diarization/ASR", "labels": "private", "license": "Private"},
        {"name": "MyST", "category": ChildVoxCategory.SPEECH, "task": "ASR", "labels": "transcript", "license": "Customized"},
        {"name": "TinyVox", "category": ChildVoxCategory.SPEECH, "task": "phoneme_ASR", "labels": "IPA/ARPABET", "license": "Not Specified"},
    ]


def balanced_distribution() -> list[dict[str, Any]]:
    """Figure 2 — ChildVox-Balanced training subset (N=64,641)."""
    return [
        {"dataset": "MyST", "samples": 10_000, "pct": 15.5},
        {"dataset": "TinyVox", "samples": 10_000, "pct": 15.5},
        {"dataset": "SpeechMaturity", "samples": 9_199, "pct": 14.2},
        {"dataset": "SpeechOcean762", "samples": 6_000, "pct": 9.3},
        {"dataset": "BabbleCor", "samples": 5_112, "pct": 7.9},
        {"dataset": "ReCANVo", "samples": 4_128, "pct": 6.4},
        {"dataset": "PERCEPT-R", "samples": 4_000, "pct": 6.2},
        {"dataset": "ICBHI-Crackles", "samples": 3_247, "pct": 5.0},
        {"dataset": "CirCor", "samples": 2_811, "pct": 4.3},
        {"dataset": "ICBHI-Wheeze", "samples": 2_674, "pct": 4.1},
        {"dataset": "CryBank", "samples": 2_088, "pct": 3.2},
        {"dataset": "AudioSet-Child", "samples": 2_000, "pct": 3.1},
        {"dataset": "ICBHI-Condition", "samples": 1_900, "pct": 2.9},
        {"dataset": "SPRSound", "samples": 1_482, "pct": 2.3},
    ]
