"""ICAD / BeSTAD anomaly-detection evaluation (Sec. IV-C2, Table III)."""

from __future__ import annotations

from typing import Sequence

from ltx_trainer.trajgenagent.aggregate_metrics import evaluate_table_iii_metrics
from ltx_trainer.trajgenagent.benchmarks import TABLE_III_ANOMALY, TABLE_V_TOOL_STABILITY
from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.corpus import flatten_corpus, reference_corpus_for_dataset
from ltx_trainer.trajgenagent.trajectory import DailyTrajectory


def anomaly_scores(dataset: str, model: str = "TrajGenAgent") -> dict[str, float]:
    row = next(r for r in TABLE_III_ANOMALY if r["dataset"] == dataset and r["model"] == model)
    return {
        "bestad_auroc": float(row["bestad_auroc"]),
        "bestad_ap": float(row["bestad_ap"]),
        "icad_visit_auroc": float(row["icad_visit_auroc"]),
        "icad_visit_ap": float(row["icad_visit_ap"]),
        "icad_ind_auroc": float(row["icad_ind_auroc"]),
    }


def closer_to_chance(score: float, target: float = 0.5) -> float:
    return abs(score - target)


def compute_anomaly_metrics(
    generated: Sequence[DailyTrajectory],
    reference: Sequence[DailyTrajectory],
) -> dict[str, float]:
    return evaluate_table_iii_metrics(generated, reference)


def anomaly_evaluation_summary(
    cfg: TrajGenAgentConfig | None = None,
    *,
    generated: Sequence[DailyTrajectory] | None = None,
    reference: Sequence[DailyTrajectory] | None = None,
) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    ref_corpus = reference_corpus_for_dataset(cfg.default_dataset, seed=7, n_users=cfg.reference_users, cfg=cfg)
    ref_trajs = reference if reference is not None else flatten_corpus(ref_corpus)
    if generated is None:
        from ltx_trainer.trajgenagent.batch_eval import generate_eval_batch

        generated, _ = generate_eval_batch(n=16, seed=11, cfg=cfg)
    live_numosim = compute_anomaly_metrics(generated, ref_trajs)
    paper_numosim = anomaly_scores("NumoSim")
    workflow = next(r for r in TABLE_V_TOOL_STABILITY if r["strategy"] == "Deterministic workflow")
    freeform = next(r for r in TABLE_V_TOOL_STABILITY if r["strategy"] == "Free-form tool calling")
    return {
        "balanced_split_pos_neg": 0.5,
        "numosim_live": live_numosim,
        "numosim_paper": paper_numosim,
        "numosim": live_numosim,
        "mobilitysyn": anomaly_scores("MobilitySyn"),
        "numosim_bestad_near_chance": closer_to_chance(live_numosim["bestad_auroc"]) < 0.15,
        "tool_stability": {
            "workflow_trajectory_success": workflow["trajectory_success"],
            "freeform_trajectory_success": freeform["trajectory_success"],
        },
        "config": cfg.__dict__,
    }
