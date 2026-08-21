"""Speech CL survey framework card and bundled tables (arXiv:2605.24863)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.speechcl.config import SpeechClConfig
from ltx_trainer.speechcl.geometry import drift_report
from ltx_trainer.speechcl.lalm_stages import hybrid_cl_consensus, stage_transitions
from ltx_trainer.speechcl.layout import LIMITATIONS
from ltx_trainer.speechcl.mitigation import mitigation_reference_map
from ltx_trainer.speechcl.mock import compare_adaptation_modes
from ltx_trainer.speechcl.open_problems import open_problems
from ltx_trainer.speechcl.taxonomy import (
    AdaptationLayer,
    GeometryEvolution,
    classical_vs_representation_centric,
    describe_adaptation_layer,
    describe_geometry,
)


def framework_card(cfg: SpeechClConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpeechClConfig()
    return {
        "name": "Speech CL Taxonomy",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "venue": cfg.venue,
        "authors": "Yang Xiao, Siyi Wang, Eun-Jung Holden, Ting Dang",
        "perspective": (
            "Continual learning for speech is about preserving and evolving "
            "shared representation geometry, not isolated task knowledge."
        ),
        "geometry_axes": [g.value for g in GeometryEvolution],
        "adaptation_layers": [a.value for a in AdaptationLayer],
        "mitigation_families": ["replay", "regularization", "architectural_isolation"],
        "foundation_models": list(cfg.foundation_models),
        "classical_cl": list(cfg.classical_cl_settings),
        "taxonomy_contrast": classical_vs_representation_centric(),
    }


def table_foundation_models() -> list[dict[str, str]]:
    """Representative speech/audio foundation models cited in §1."""
    return [
        {"model": "wav2vec 2.0", "role": "self-supervised speech representations", "year": "2020"},
        {"model": "HuBERT", "role": "masked prediction of hidden units", "year": "2021"},
        {"model": "Whisper", "role": "weakly supervised large-scale ASR", "year": "2022"},
        {"model": "Qwen2-Audio", "role": "audio-language instruction model", "year": "2024"},
        {"model": "LALMs", "role": "large audio-language multimodal reasoning", "year": "2025+"},
    ]


def table_geometry_taxonomy() -> list[dict[str, Any]]:
    return [describe_geometry(g) for g in GeometryEvolution]


def table_adaptation_layers() -> list[dict[str, Any]]:
    return [describe_adaptation_layer(layer) for layer in AdaptationLayer]


def headline_findings() -> dict[str, Any]:
    return {
        "central_claim": (
            "Classical task/domain/class-incremental taxonomies are insufficient; "
            "CL in speech must track geometry preservation, expansion, alignment, and specialization."
        ),
        "lalm_insight": "Standard LALM post-training is implicit multimodal continual learning (Fig. 1)",
        "hybrid_norm": hybrid_cl_consensus()["observation"],
        "open_problem_count": len(open_problems()),
        "privacy_constraint": "Raw-audio replay limited by biometric voice privacy",
    }


def evaluation_demo(cfg: SpeechClConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpeechClConfig()
    return {
        "paper": cfg.paper_arxiv,
        "geometry_drift": drift_report(),
        "adaptation_comparison": compare_adaptation_modes(),
        "lalm_transitions": len(stage_transitions()),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "geometry_taxonomy": table_geometry_taxonomy(),
        "adaptation_layers": table_adaptation_layers(),
        "foundation_models": table_foundation_models(),
        "mitigation_reference_map": mitigation_reference_map(),
        "lalm_stage_transitions": stage_transitions(),
        "hybrid_cl_consensus": hybrid_cl_consensus(),
        "open_problems": open_problems(),
        "headlines": headline_findings(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_findings(),
    }
