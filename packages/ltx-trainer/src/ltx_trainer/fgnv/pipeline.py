"""Fine-grained NV TTS framework card and paper tables (arXiv:2605.25504)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fgnv.config import FgnvConfig
from ltx_trainer.fgnv.layout import LIMITATIONS
from ltx_trainer.fgnv.mock import circumplex_embedding, demo_recognition
from ltx_trainer.fgnv.parsers import encode_utterance


def framework_card(cfg: FgnvConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FgnvConfig()
    return {
        "name": "Fine-Grained Non-Verbal Emotional TTS",
        "paper": cfg.paper_arxiv,
        "demo": cfg.demo_page,
        "idea": (
            "EARS-derived fine-grained NV annotations (style, frequency, duration) "
            "with Grad-TTS + arousal/valence emotion encoder and HiFi-GAN vocoder."
        ),
        "emotions_evaluated": list(cfg.emotions),
        "nv_types": [
            "laughter-open",
            "laughter-closed",
            "cheering",
            "yelling",
            "crying",
            "screaming",
        ],
        "designs_evaluated": [
            "Only Verbal",
            "Verbal + Coarse-grained NV (NVTTS)",
            "Verbal + Fine-Grained NV (proposed)",
        ],
        "dataset": {
            "nv_utterances": cfg.nv_utterances,
            "verbal_mix_hours": cfg.mixed_verbal_hours,
            "female_speakers": cfg.female_speakers,
        },
        "training": {"backbone": cfg.backbone, "vocoder": cfg.vocoder, "iterations": cfg.training_iterations},
        "defaults": cfg.__dict__,
    }


def table_i_nv_counts() -> list[dict[str, Any]]:
    """Table 1 — NV category transcripts and utterance counts."""
    return [
        {"category": "Cheering", "transcript_examples": ["Wo ho", "Yo"], "type_count": 2, "utterance_count": 262},
        {"category": "Yelling", "transcript_examples": ["Hey"], "type_count": 1, "utterance_count": 328},
        {"category": "Laughter-open", "transcript_examples": ["Ha"], "type_count": 1, "utterance_count": 266},
        {"category": "Laughter-closed", "transcript_examples": ["Ha"], "type_count": 1, "utterance_count": 220},
        {"category": "Crying", "transcript_examples": ["Whep", "Wuu", "Sneeze"], "type_count": 3, "utterance_count": 230},
        {"category": "Screaming", "transcript_examples": ["Ah"], "type_count": 1, "utterance_count": 154},
    ]


def table_ii_annotation_comparison() -> dict[str, list[str]]:
    """Table 2 — coarse vs fine-grained annotation."""
    return {
        "coarse_grained": ["<crying>", "Simple style-level tag"],
        "fine_grained": [
            '<(crying) wuuuuu whep>',
            "Style-level tag",
            "Vocalization type and control",
            "Frequency and duration control",
        ],
    }


def figure_ii_overall_metrics() -> list[dict[str, Any]]:
    """Fig. 2 — nMOS, eMOS, emotion recognition accuracy."""
    return [
        {"design": "Only Verbal", "nmos": 3.54, "emos": None, "recognition_accuracy_pct": 65.5},
        {"design": "Verbal + Coarse-grained NV", "nmos": None, "emos": None, "recognition_accuracy_pct": None},
        {"design": "Verbal + Fine-Grained NV", "nmos": None, "emos": 4.20, "recognition_accuracy_pct": 78.8},
    ]


def table_iii_per_emotion_mos() -> dict[str, dict[str, dict[str, float]]]:
    """Table 3 — nMOS and eMOS per emotion (paper values)."""
    return {
        "nmos": {
            "Only Verbal": {"happy": 3.67, "sad": 3.69, "anger": 3.61, "fear": 3.19},
            "Coarse-grained NV": {"happy": 3.19, "sad": 3.73, "anger": None, "fear": None},
            "Fine-Grained NV": {"happy": 3.43, "sad": 3.67, "anger": 3.34, "fear": 3.18},
        },
        "emos": {
            "Only Verbal": {"happy": 3.78, "sad": 3.83, "anger": 3.74, "fear": 3.89},
            "Coarse-grained NV": {"happy": 3.85, "sad": 4.15, "anger": None, "fear": None},
            "Fine-Grained NV": {"happy": 4.21, "sad": 4.25, "anger": 4.06, "fear": 4.28},
        },
    }


def emotion_recognition_accuracy() -> dict[str, float]:
    """§5.3.1 per-emotion accuracy (Fine-Grained NV design)."""
    return {
        "happy": 82.5,
        "sad": 98.3,
        "anger": 64.3,
        "fear": 82.7,
        "average": 78.8,
    }


def figure_iv_preference() -> dict[str, list[dict[str, Any]]]:
    """Fig. 4 — preference rankings for happy/sad NV variants."""
    return {
        "happy": [
            {"expression": '<(cheering) Wo ho>', "note": "top preference"},
            {"expression": '<(cheering) Yo>', "note": "top preference"},
            {"expression": '<(Laughter-open) ha ha>', "note": "lower preference"},
            {"expression": '<(Laughter-closed) ha ha>', "note": "62% ranked third"},
        ],
        "sad": [
            {"expression": '<(crying) wuuuuuuu whep>', "note": "56% ranked first"},
            {"expression": '<(crying) whep>', "note": "lower preference"},
            {"expression": '<(crying) sneeze>', "note": "lower preference"},
            {"expression": '<(crying) wuuuuuuu>', "note": "56% ranked fourth"},
        ],
    }


def headline_results() -> dict[str, Any]:
    return {
        "emos_fine_grained": 4.20,
        "recognition_accuracy_pct": 78.8,
        "recognition_gain_vs_verbal_only_pct": 13.3,
        "nmos_tradeoff": "Only Verbal highest nMOS 3.54; NV adds expressiveness at minor naturalness cost",
        "emotion_accuracy": emotion_recognition_accuracy(),
        "preference": {
            "happy": "cheering preferred over laughter",
            "sad": "multi-part cry wuuuuuuu whep preferred",
        },
    }


def evaluation_demo(cfg: FgnvConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FgnvConfig()
    sample = "<(crying) wuuuuu whep> why you do this to me."
    enc = encode_utterance(sample)
    emb = circumplex_embedding(arousal=-0.6, valence=-0.8)
    rec = demo_recognition(sample, arousal=-0.6, valence=-0.8)
    laugh = encode_utterance("<(Laughter-open) ha ha ha> what did you do")
    return {
        "sample_encoding": enc,
        "emotion_embedding_dim": len(emb),
        "demo_recognition": rec,
        "laughter_discrete_count": laugh["discrete_count"],
        "total_nv_utterances": cfg.nv_utterances,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "table_i_nv_counts": table_i_nv_counts(),
        "table_ii_annotation": table_ii_annotation_comparison(),
        "figure_ii_metrics": figure_ii_overall_metrics(),
        "table_iii_per_emotion": table_iii_per_emotion_mos(),
        "emotion_recognition_accuracy": emotion_recognition_accuracy(),
        "figure_iv_preference": figure_iv_preference(),
        "headlines": headline_results(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
