"""Mock MMAE evaluation runs reproducing rubric scoring pipeline."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mmae.benchmarks import TABLE2_OVERALL
from ltx_trainer.mmae.catalog import synthetic_catalog
from ltx_trainer.mmae.config import MMAEConfig
from ltx_trainer.mmae.metrics import aggregate_rates_pct, rates_from_rubric_scores
from ltx_trainer.mmae.rubrics import score_sample


def model_skill_from_table(model: str) -> tuple[float, float]:
    row = TABLE2_OVERALL[model]
    return row["IFR"] / 100.0, row["CR"] / 100.0


def evaluate_model_on_catalog(
    model: str,
    *,
    cfg: MMAEConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = cfg or MMAEConfig()
    p_if, p_cr = model_skill_from_table(model)
    samples = synthetic_catalog()
    per_sample: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        scored = score_sample(sample, p_correct_if=p_if, p_correct_cr=p_cr, cfg=cfg, seed=seed + i)
        rates = rates_from_rubric_scores(sample, scored["rubrics"])
        per_sample.append({"sample_id": sample.sample_id, **rates, "scored": scored})
    agg = aggregate_rates_pct([{k: v for k, v in s.items() if k in ("IFR", "CR", "EMR")} for s in per_sample])
    return {"model": model, "rates_pct": agg, "samples": per_sample}


def full_eval_demo(cfg: MMAEConfig | None = None, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or MMAEConfig()
    models = ("Audio-Omni", "Step-Audio-EditX", "Identity", "Noise")
    return {
        "catalog": synthetic_catalog()[0].to_dict(),
        "catalog_summary": {s.sample_id: s.num_rubrics for s in synthetic_catalog()},
        "model_runs": {
            m: evaluate_model_on_catalog(m, cfg=cfg, seed=seed)["rates_pct"] for m in models
        },
        "judger": cfg.judger,
        "majority_vote": {
            "votes": cfg.params.majority_votes,
            "threshold": cfg.params.majority_threshold,
        },
    }
