"""Kinematics ablation with live Table VI metrics (Sec. IV-I)."""

from __future__ import annotations

from ltx_trainer.trajgenagent.aggregate_metrics import evaluate_table_ii_metrics
from ltx_trainer.trajgenagent.benchmarks import TABLE_VI_KINEMATICS, TABLE_VII_KINEMATICS_ANOMALY
from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.corpus import flatten_corpus, reference_corpus_for_dataset
from ltx_trainer.trajgenagent.metrics import paper_table_ii_row
from ltx_trainer.trajgenagent.orchestrator import synthesize_activity_chain_from_profile
from ltx_trainer.trajgenagent.profiles import build_profile_from_activity_chains
from ltx_trainer.trajgenagent.workflow import generate_daily_trajectory, run_worker_workflow


def paper_kinematics_row(dataset: str, *, with_kinematics: bool) -> dict[str, float]:
    label = "with_kinematics" if with_kinematics else "without_kinematics"
    row = next(r for r in TABLE_VI_KINEMATICS if r["dataset"] == dataset and r["variant"] == label)
    return {k: float(row[k]) for k in row if k not in {"dataset", "variant"}}


def paper_anomaly_ablation_row(dataset: str, *, with_kinematics: bool) -> dict[str, float]:
    label = "with_kinematics" if with_kinematics else "without_kinematics"
    row = next(r for r in TABLE_VII_KINEMATICS_ANOMALY if r["dataset"] == dataset and r["variant"] == label)
    return {k: float(row[k]) for k in row if k not in {"dataset", "variant"}}


def run_kinematics_ablation(
    *,
    dataset: str = "NumoSim",
    seed: int = 42,
    cfg: TrajGenAgentConfig | None = None,
) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    ref = flatten_corpus(reference_corpus_for_dataset(dataset, seed=seed, n_users=8, cfg=cfg))
    profile = build_profile_from_activity_chains("u_ablation", [("Home", "Work", "EatOut", "Work", "Home")])
    chain = synthesize_activity_chain_from_profile(profile=profile, cfg=cfg)
    with_cfg = cfg
    without_cfg = TrajGenAgentConfig(**{**cfg.__dict__, "use_kinematics": False})
    with_kin = run_worker_workflow(chain, seed=seed, cfg=with_cfg, profile=profile)
    without_kin = run_worker_workflow(chain, seed=seed, cfg=without_cfg, profile=profile)
    gen_with = tuple(
        generate_daily_trajectory(seed=seed + i, cfg=with_cfg, use_peer_pool=False)["stage2"]["trajectory"]
        for i in range(8)
    )
    gen_without = tuple(
        generate_daily_trajectory(seed=seed + 100 + i, cfg=without_cfg, use_peer_pool=False)["stage2"]["trajectory"]
        for i in range(8)
    )
    live_with = evaluate_table_ii_metrics(gen_with, ref, dataset=dataset, cfg=with_cfg)
    live_without = evaluate_table_ii_metrics(gen_without, ref, dataset=dataset, cfg=without_cfg)
    return {
        "dataset": dataset,
        "with_kinematics": {
            "trajectory_success": with_kin["trajectory_success"],
            "live_table_ii": live_with,
            "paper_anchors": paper_table_ii_row(dataset),
        },
        "without_kinematics": {
            "trajectory_success": without_kin["trajectory_success"],
            "live_table_ii": live_without,
            "paper_anchors": paper_kinematics_row(dataset, with_kinematics=False),
        },
        "kinematics_improves_distance": live_with["distance"] <= live_without["distance"],
        "paper_table_vi": TABLE_VI_KINEMATICS,
        "paper_table_vii": TABLE_VII_KINEMATICS_ANOMALY,
    }


def ablation_summary(cfg: TrajGenAgentConfig | None = None) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    numosim = run_kinematics_ablation(dataset="NumoSim", cfg=cfg)
    mobilitysyn = run_kinematics_ablation(dataset="MobilitySyn", cfg=cfg)
    n_bestad = paper_anomaly_ablation_row("NumoSim", with_kinematics=True)
    n_bestad_no = paper_anomaly_ablation_row("NumoSim", with_kinematics=False)
    return {
        "numosim": numosim,
        "mobilitysyn": mobilitysyn,
        "numosim_bestad_with_kin_closer_to_chance": abs(n_bestad["bestad_auroc"] - 0.5) < abs(
            n_bestad_no["bestad_auroc"] - 0.5
        ),
        "kinematics_improves_icad_on_numosim": abs(n_bestad["icad_visit_ap"] - 0.5)
        < abs(n_bestad_no["icad_visit_ap"] - 0.5),
        "live_kinematics_improves_distance_numosim": numosim["kinematics_improves_distance"],
    }
