"""Framework card, evaluation demo, smoke (arXiv:2606.07229)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mmae.benchmarks import benchmarks_bundle, paper_claims, table1_statistics
from ltx_trainer.mmae.catalog import appendix_b_dog_extraction, catalog_summary, synthetic_catalog
from ltx_trainer.mmae.config import MMAEConfig
from ltx_trainer.mmae.constants import MMAE_GITHUB, MMAE_HUB_DATASET, MMAE_JUDGER
from ltx_trainer.mmae.editors import editors_card
from ltx_trainer.mmae.judger import JUDGER_SYSTEM_PROMPT, format_user_prompt
from ltx_trainer.mmae.layout import LIMITATIONS
from ltx_trainer.mmae.sample import MMAESample
from ltx_trainer.mmae.simulation import full_eval_demo
from ltx_trainer.mmae.taxonomy import taxonomy_summary


def framework_card(cfg: MMAEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MMAEConfig()
    sample = appendix_b_dog_extraction()
    return {
        "name": "MMAE",
        "paper": cfg.paper_arxiv,
        "title": cfg.paper_title,
        "task": "Massive multitask instruction-based audio editing benchmark",
        "github": MMAE_GITHUB,
        "hub_dataset": MMAE_HUB_DATASET,
        "judger": MMAE_JUDGER,
        "statistics": table1_statistics(),
        "taxonomy": taxonomy_summary(),
        "evaluation": {
            "dimensions": ["Instruction Following (IFR)", "Consistency (CR)", "Exact Match (EMR)"],
            "paradigm": "Rubric-based multiple-choice with majority vote (2/3)",
            "rubric_principles": taxonomy_summary()["rubric_principles"],
        },
        "example_sample_id": sample.sample_id,
        "example_instruction": sample.instruction[:120] + "...",
        "candidate_models": list(cfg.candidate_models),
        "editors": editors_card(),
    }


def paper_limitations() -> list[str]:
    return list(LIMITATIONS)


def evaluation_demo(cfg: MMAEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MMAEConfig()
    sample = appendix_b_dog_extraction()
    rubric = sample.rubrics[0]
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "benchmarks": benchmarks_bundle(),
        "catalog_summary": catalog_summary(),
        "judger_system_prompt_excerpt": JUDGER_SYSTEM_PROMPT[:200] + "...",
        "judger_user_prompt_example": format_user_prompt(
            rubric.question,
            rubric.all_choices(),
        ),
        "simulation": full_eval_demo(cfg, seed=0),
    }


def evaluation_smoke(cfg: MMAEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MMAEConfig()
    demo = evaluation_demo(cfg)
    claims = paper_claims()
    t1 = demo["benchmarks"]["table1"]
    sample = appendix_b_dog_extraction()

    assert t1["total_samples"] == 2000
    assert t1["total_rubrics"] == 17_741
    assert claims["all_emr_under_5pct"]
    assert claims["identity_high_cr"]
    assert claims["complexity_degrades_audio_omni"]
    assert sample.num_rubrics >= 4
    assert MMAESample.from_dict(sample.to_dict()).sample_id == sample.sample_id

    sim = demo["simulation"]["model_runs"]
    assert sim["Identity"]["CR"] > sim["Identity"]["IFR"]
    assert sim["Noise"]["CR"] < 30.0
    assert "Identity" in demo["framework"]["editors"]["local_baselines"]

    return {
        "status": "ok",
        "paper": cfg.paper_arxiv,
        "total_samples": t1["total_samples"],
        "total_rubrics": t1["total_rubrics"],
        "all_emr_under_5pct": claims["all_emr_under_5pct"],
        "demo_keys": list(demo.keys()),
    }
