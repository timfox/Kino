"""MTAVG-Bench 2.0 framework card, knowledge, and evaluation demos."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mtavg2.benchmarks import benchmarks_bundle
from ltx_trainer.mtavg2.config import MTAVG2Config
from ltx_trainer.mtavg2.diagnosis import diagnose_clip_proxy, score_generation_vs_bench
from ltx_trainer.mtavg2.scoring import (
    primary_issue_accuracy,
    rationale_consistency_pct,
    score_multiple_choice,
    score_single_choice,
    sub_dimension_score,
    temporal_localization_accuracy,
    weighted_average_sub_dims,
)
from ltx_trainer.mtavg2.taxonomy import taxonomy_card


def framework_card(cfg: MTAVG2Config | None = None) -> dict[str, Any]:
    cfg = cfg or MTAVG2Config()
    return {
        "name": "MTAVG-Bench 2.0",
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "huggingface": cfg.huggingface_dataset,
        "problem": (
            "Diagnose high-level cinematic expressiveness failures in multi-talker audio-video "
            "generation—beyond lip-sync and low-level A-V alignment."
        ),
        "taxonomy": {
            "categories": list(cfg.category_codes),
            "sub_dimensions": list(cfg.sub_dimension_codes),
            "n_failure_modes": cfg.n_failure_modes,
        },
        "dataset": {
            "n_videos": cfg.n_videos,
            "n_qa": cfg.n_qa_instances,
            "segment_s": cfg.segment_duration_s,
            "question_formats": list(cfg.question_formats),
        },
        "annotation_qa": {
            "expert_pool": cfg.expert_pool_size,
            "two_expert_agreement": cfg.two_expert_agreement_rate,
            "cohens_kappa": cfg.cohens_kappa,
        },
        "ltx_paper_finding": (
            "LTX 2.3 shows competitive Desync (lowest in Table 4) but elevated scene-level "
            "failure rates on dialogue, interaction, mood, grammar, and continuity (Fig. 4)."
        ),
    }


def knowledge_card(cfg: MTAVG2Config | None = None) -> dict[str, Any]:
    cfg = cfg or MTAVG2Config()
    tax = taxonomy_card()
    bench = benchmarks_bundle()
    return {
        "framework": framework_card(cfg),
        "taxonomy_summary": tax,
        "key_results": {
            "best_omni_diagnoser": "Gemini 3.1 Pro",
            "gemini31_pro_avg_pct": bench["gemini31_pro_avg_diagnosis"],
            "temporal_pia_pct": cfg.gemini31_pro_pia_pct,
            "temporal_tla_pct": cfg.gemini31_pro_tla_pct,
            "temporal_rc_pct": cfg.gemini31_pro_rc_pct,
            "ltx23_table4": bench["ltx23_holistic"],
        },
        "integration": {
            "env_diag": "GOPEX_MTAVG2_DIAG=1",
            "env_fold": "GOPEX_MTAVG2_FOLD=1 or GOPEX_AV_FOLD_HOOKS=...,mtavg2",
            "script_suffix": "ltx_trainer.mtavg2.diagnosis.ltx_script_prompt_suffix",
            "cli": "./scripts/gopex-mtavg2.sh",
        },
    }


def evaluation_demo(cfg: MTAVG2Config | None = None) -> dict[str, Any]:
    cfg = cfg or MTAVG2Config()
    bench = benchmarks_bundle()

    # Scoring smoke
    mcq = score_single_choice("interaction_eyeline_failure", "interaction_eyeline_failure")
    multi = score_multiple_choice(
        ["shot_progression_issues", "spatial_continuity_break"],
        ["shot_progression_issues", "spatial_continuity_break", "action_continuity_break"],
    )
    sub_scores = {"DP": sub_dimension_score([0.8, 0.6, 1.0]), "CT": sub_dimension_score([0.5, 0.7])}
    avg = weighted_average_sub_dims(sub_scores, {"DP": 3, "CT": 2})
    pia = primary_issue_accuracy(["speech_mode_confusion"], ["speech_mode_confusion"])
    tla = temporal_localization_accuracy([{"0-2s", "2-4s"}], [{"0-2s", "2-4s", "4-6s"}])
    rc = rationale_consistency_pct([4.0, 5.0, 3.0])

    ltx_clip = score_generation_vs_bench(
        audio_aesthetic=3.9,
        lip_sync=0.45,
        av_align=0.12,
        desync=0.38,
        ta_align=0.17,
        tv_align=0.19,
        latent_temporal_diff=0.42,
    )
    diag = diagnose_clip_proxy(sub_dim="IP", latent_temporal_diff=0.5, latent_spatial_std=0.6)

    return {
        "framework": framework_card(cfg),
        "scoring_smoke": {
            "mcq_exact_match": mcq,
            "multi_choice_coverage": multi,
            "weighted_sub_dim_avg": round(avg, 4),
            "pia": pia,
            "tla": tla,
            "rc_pct": round(rc, 2),
        },
        "ltx_generation_smoke": ltx_clip,
        "sample_diagnosis": diag,
        "paper_tables": bench,
        "insights": {
            "metric_gap": (
                "Grok leads several Table-4 audio metrics yet still shows high diagnostic "
                "failure rates—motivates MTAVG-Bench 2.0 over fidelity-only gates."
            ),
            "ltx_strength": "LTX 2.3 lowest Desync in Table 4; weak on scene coordination dimensions.",
            "omni_ceiling": "Gemini 3.1 Pro ~62% avg diagnosis; temporal PIA/TLA ~61% vs RC ~84%.",
        },
    }


def evaluation_smoke(cfg: MTAVG2Config | None = None) -> dict[str, Any]:
    ev = evaluation_demo(cfg)
    return {
        "ok": ev["scoring_smoke"]["mcq_exact_match"] == 1.0,
        "gemini31_pro_avg": ev["paper_tables"]["gemini31_pro_avg_diagnosis"],
        "ltx_expressiveness_risk": ev["ltx_generation_smoke"]["cinematic_diagnosis"]["severity_proxy"],
    }
