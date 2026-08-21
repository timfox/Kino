"""Reference / synthetic trajectory corpora for eval and ICL."""

from __future__ import annotations

import random

from ltx_trainer.trajgenagent.config import TrajGenAgentConfig, NUMOSIM_ACTIVITIES
from ltx_trainer.trajgenagent.orchestrator import synthesize_activity_chain_from_profile
from ltx_trainer.trajgenagent.profiles import IndividualProfile, build_profile_from_activity_chains, build_profile_from_trajectories
from ltx_trainer.trajgenagent.trajectory import DailyTrajectory


def _default_chains(seed: int) -> list[tuple[str, ...]]:
    rng = random.Random(seed)
    templates = [
        ("Home", "Work", "EatOut", "Work", "Home"),
        ("Home", "Work", "Shop", "Work", "Errand", "Home"),
        ("Home", "Work", "EatOut", "Work", "Leisure", "Home"),
    ]
    return [templates[rng.randint(0, len(templates) - 1)] for _ in range(3)]


def build_synthetic_corpus(
    *,
    seed: int = 42,
    n_users: int = 6,
    cfg: TrajGenAgentConfig | None = None,
) -> dict[str, tuple[DailyTrajectory, ...]]:
    from ltx_trainer.trajgenagent.workflow import run_worker_workflow

    cfg = cfg or TrajGenAgentConfig()
    users: dict[str, tuple[DailyTrajectory, ...]] = {}
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    for i in range(n_users):
        uid = f"u_{i:03d}"
        chains = _default_chains(seed + i)
        profile = build_profile_from_activity_chains(uid, chains)
        trajs: list[DailyTrajectory] = []
        for j, wd in enumerate(weekdays):
            chain = synthesize_activity_chain_from_profile(
                weekday=wd,
                day_type="weekday",
                profile=profile,
                rng=random.Random(seed + i * 10 + j),
                cfg=cfg,
            )
            stage2 = run_worker_workflow(
                chain,
                individual_id=uid,
                weekday=wd,
                seed=seed + i * 10 + j,
                cfg=cfg,
                profile=profile,
            )
            trajs.append(stage2["trajectory"])
        users[uid] = tuple(trajs)
    return users


def reference_corpus_for_dataset(
    dataset: str = "NumoSim",
    *,
    seed: int = 0,
    n_users: int = 24,
    cfg: TrajGenAgentConfig | None = None,
) -> dict[str, tuple[DailyTrajectory, ...]]:
    cfg = cfg or TrajGenAgentConfig()
    return build_synthetic_corpus(
        seed=seed + (1 if dataset == "MobilitySyn" else 0),
        n_users=n_users,
        cfg=cfg,
    )


def flatten_corpus(corpus: dict[str, tuple[DailyTrajectory, ...]]) -> tuple[DailyTrajectory, ...]:
    out: list[DailyTrajectory] = []
    for trajs in corpus.values():
        out.extend(trajs)
    return tuple(out)


def profile_for_individual(
    corpus: dict[str, tuple[DailyTrajectory, ...]],
    individual_id: str,
) -> IndividualProfile:
    if individual_id in corpus:
        return build_profile_from_trajectories(individual_id, corpus[individual_id])
    fallback = next(iter(corpus.values()))
    return build_profile_from_trajectories(individual_id, fallback)


def activity_vocabulary(dataset: str = "NumoSim") -> tuple[str, ...]:
    return NUMOSIM_ACTIVITIES if dataset == "NumoSim" else (
        "Home",
        "Work",
        "EatOut",
        "Shop",
        "Leisure",
        "Errand",
    )
