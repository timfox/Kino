"""Batch trajectory generation + live Table II / III evaluation."""

from __future__ import annotations

from ltx_trainer.trajgenagent.aggregate_metrics import compare_to_reference, evaluate_table_ii_metrics
from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.corpus import flatten_corpus, reference_corpus_for_dataset
from ltx_trainer.trajgenagent.metrics import paper_table_ii_row
from ltx_trainer.trajgenagent.trajectory import DailyTrajectory
from ltx_trainer.trajgenagent.workflow import generate_daily_trajectory


def generate_eval_batch(
    *,
    n: int = 16,
    seed: int = 0,
    cfg: TrajGenAgentConfig | None = None,
    use_peer_pool: bool = True,
) -> tuple[tuple[DailyTrajectory, ...], int]:
    cfg = cfg or TrajGenAgentConfig()
    ref_corpus = reference_corpus_for_dataset(
        cfg.default_dataset,
        seed=seed,
        n_users=min(cfg.reference_users, 12),
        cfg=cfg,
    )
    out: list[DailyTrajectory] = []
    successes = 0
    for i in range(n):
        result = generate_daily_trajectory(
            individual_id=f"u_{i % 12:03d}",
            weekday=("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")[i % 5],
            seed=seed + i,
            cfg=cfg,
            use_peer_pool=use_peer_pool,
            reference_corpus=ref_corpus,
        )
        traj = result["stage2"]["trajectory"]
        out.append(traj)
        if result["stage2"]["trajectory_success"] == 1.0:
            successes += 1
    return tuple(out), successes


def batch_generation_eval(
    *,
    n: int = 16,
    seed: int = 0,
    dataset: str = "NumoSim",
    use_peer_pool: bool = True,
    cfg: TrajGenAgentConfig | None = None,
) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    cfg = TrajGenAgentConfig(**{**cfg.__dict__, "default_dataset": dataset})
    ref_corpus = reference_corpus_for_dataset(dataset, seed=seed, n_users=cfg.reference_users, cfg=cfg)
    reference = flatten_corpus(ref_corpus)
    generated, successes = generate_eval_batch(n=n, seed=seed, cfg=cfg, use_peer_pool=use_peer_pool)
    live_ii = evaluate_table_ii_metrics(generated, reference, dataset=dataset, cfg=cfg)
    paper = paper_table_ii_row(dataset)
    geo = paper_table_ii_row(dataset, model="Geo-Llama")
    comparison = compare_to_reference(generated, reference, dataset=dataset, cfg=cfg)
    return {
        "dataset": dataset,
        "n": n,
        "use_peer_pool": use_peer_pool,
        "trajectory_success_rate": successes / n,
        "live_table_ii": live_ii,
        "paper_trajgenagent": paper,
        "paper_geo_llama": geo,
        "beats_geo_llama_distance_jsd": paper["distance"] < geo["distance"],
        "beats_geo_llama_g_radius": paper["g_radius"] < geo["g_radius"],
        "live_beats_geo_distance": live_ii["distance"] < geo["distance"],
        "comparison": comparison,
    }


def dual_dataset_batch_summary(
    *,
    n: int = 12,
    seed: int = 0,
    cfg: TrajGenAgentConfig | None = None,
) -> dict[str, object]:
    numosim = batch_generation_eval(n=n, seed=seed, dataset="NumoSim", cfg=cfg)
    mobilitysyn = batch_generation_eval(n=n, seed=seed + 1000, dataset="MobilitySyn", cfg=cfg)
    return {
        "numosim": numosim,
        "mobilitysyn": mobilitysyn,
        "both_beat_geo_llama_distance": numosim["beats_geo_llama_distance_jsd"]
        and mobilitysyn["beats_geo_llama_distance_jsd"],
    }
