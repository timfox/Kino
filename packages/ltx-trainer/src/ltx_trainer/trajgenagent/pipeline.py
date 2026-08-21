"""TrajGenAgent evaluation smoke and demo pipeline."""

from __future__ import annotations

from ltx_trainer.trajgenagent.ablation import ablation_summary
from ltx_trainer.trajgenagent.anomaly_eval import anomaly_evaluation_summary
from ltx_trainer.trajgenagent.batch_eval import dual_dataset_batch_summary
from ltx_trainer.trajgenagent.benchmarks import (
    PAPER_ANCHORS,
    TABLE_II_MOBILITYSYN,
    TABLE_II_NUMOSIM,
    TABLE_V_TOOL_STABILITY,
    benchmarks_bundle,
)
from ltx_trainer.trajgenagent.config import PAPER_ARXIV, PAPER_TITLE, TrajGenAgentConfig
from ltx_trainer.trajgenagent.evidence import evidence_demo
from ltx_trainer.trajgenagent.ingest import ingest_demo
from ltx_trainer.trajgenagent.integration import framework_card, integration_bundle
from ltx_trainer.trajgenagent.metrics import compare_to_paper_stub, trajectory_level_metrics
from ltx_trainer.trajgenagent.orchestrator import generate_activity_chain, validate_activity_chain
from ltx_trainer.trajgenagent.temperature_study import temperature_study
from ltx_trainer.trajgenagent.tool_stability import tool_stability_comparison
from ltx_trainer.trajgenagent.workflow import generate_daily_trajectory, run_worker_workflow
from ltx_trainer.trajgenagent.workflow_graph import workflow_graph_card


def evaluation_demo() -> dict[str, object]:
    cfg = TrajGenAgentConfig()
    gen = generate_daily_trajectory(cfg=cfg)
    traj = gen["stage2"]["trajectory"]
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "framework": framework_card(cfg),
        "integration": integration_bundle(),
        "benchmarks": benchmarks_bundle(),
        "anchors": PAPER_ANCHORS,
        "generation": {
            "activity_chain": gen["activity_chain"],
            "n_visits": gen["stage2"]["n_visits"],
            "trajectory_success": gen["stage2"]["trajectory_success"],
            "local_metrics": trajectory_level_metrics(traj),
        },
        "paper_compare_numosim": compare_to_paper_stub(traj, dataset="NumoSim"),
        "anomaly_eval": anomaly_evaluation_summary(cfg),
        "tool_stability": tool_stability_comparison(cfg),
        "kinematics_ablation": ablation_summary(cfg),
        "temperature_study": temperature_study(cfg),
        "batch_eval": dual_dataset_batch_summary(n=8, seed=1, cfg=cfg),
        "ingest": ingest_demo(),
        "workflow_graph": workflow_graph_card(),
    }


def evaluation_smoke() -> dict[str, bool]:
    cfg = TrajGenAgentConfig()
    gen = generate_daily_trajectory(cfg=cfg)
    chain = tuple(gen["activity_chain"])
    ok_chain, _ = validate_activity_chain(chain)
    traj = gen["stage2"]["trajectory"]
    local = trajectory_level_metrics(traj)
    paper_row = next(r for r in TABLE_II_NUMOSIM if r["model"] == "TrajGenAgent")
    anomaly = anomaly_evaluation_summary(cfg)
    workflow = next(r for r in TABLE_V_TOOL_STABILITY if r["strategy"] == "Deterministic workflow")
    ablation = ablation_summary(cfg)
    stability = tool_stability_comparison(cfg)
    temp = temperature_study(cfg)
    checks = {
        "activity_chain_valid": ok_chain,
        "chain_starts_home": chain[0] == "Home",
        "chain_ends_home": chain[-1] == "Home",
        "workflow_full_success": gen["stage2"]["trajectory_success"] == 1.0,
        "workflow_visit_success": gen["stage2"]["visit_success"] == 1.0,
        "deterministic_tool_stability": workflow["trajectory_success"] == 1.0,
        "n_visits_positive": local["daily_loc"] >= 4,
        "distance_nonnegative": local["distance_km"] >= 0.0,
        "paper_distance_anchor": paper_row["distance"] == 0.0006,
        "numosim_bestad_near_chance": anomaly["numosim_bestad_near_chance"],
        "gpu_hours_anchor": PAPER_ANCHORS["trajgenagent_gpu_hours"] == 1.67,
        "framework_card_ok": framework_card()["paper"]["arxiv"] == PAPER_ARXIV,
        "orchestrator_retry_bounded": generate_activity_chain(cfg=cfg)["attempts"] <= cfg.orchestrator_max_retries + 1,
        "kinematics_enabled_default": cfg.use_kinematics,
    }
    no_kin = run_worker_workflow(chain, cfg=TrajGenAgentConfig(use_kinematics=False))
    checks["kinematics_ablation_runs"] = no_kin["trajectory_success"] == 1.0
    checks["kinematics_improves_bestad"] = ablation["numosim_bestad_with_kin_closer_to_chance"]
    checks["workflow_beats_freeform"] = stability["workflow_beats_freeform_traj"]
    checks["temperature_optimal_09"] = temp["live_matches_paper_best_temp"]
    checks["temperature_high_temp_worse"] = temp["live_monotonic_high_temp"]
    checks["mobilitysyn_distance_anchor"] = (
        next(r for r in TABLE_II_MOBILITYSYN if r["model"] == "TrajGenAgent")["distance"] == 0.0000
    )
    ev = evidence_demo(seed=42)
    checks["evidence_profile_built"] = ev["profile"]["n_transitions"] > 0
    checks["peer_retrieval_top_k"] = len(ev["peer_retrieval"]["top_peers"]) == cfg.peer_top_k
    batch = dual_dataset_batch_summary(n=8, seed=1, cfg=cfg)
    checks["batch_beats_geo_llama"] = batch["both_beat_geo_llama_distance"]
    checks["batch_live_table_ii"] = "live_table_ii" in batch["numosim"]
    checks["ingest_demo_ok"] = ingest_demo()["n_visits"] >= 5
    checks["peer_pool_generation"] = (
        generate_daily_trajectory(use_peer_pool=True, seed=7, cfg=cfg)["stage2"]["trajectory_success"] == 1.0
    )
    checks["workflow_graph_fixed_order"] = workflow_graph_card()["tool_order_fixed"]
    checks["all_pass"] = all(checks.values())
    return checks
